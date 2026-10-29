"""Independent grid coding, collisions, checkpoint resume and standalone decode."""
import copy
import json
import zipfile
from arguments import Config
from utils.formats import CHECKPOINT_FORMAT, ARCHIVE_FORMAT
import tempfile
import unittest
from pathlib import Path

import torch
from gaussian_renderer import render
from scene.dataset_readers import Camera
from smolgs.codec import encode_model, decode_model, export_reference
from smolgs.coordinate_grid import encode_coordinates, decode_coordinates
from test_ecvq import model
from utils.checkpoint import save_checkpoint, load_checkpoint


class CoordinateGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_current_formats_and_positive_depth_are_required(self):
        with self.assertRaises(ValueError):
            Config(coordinate_depth=0)
        self.assertEqual(Config().coordinate_depth, 16)
        with tempfile.TemporaryDirectory() as tmp:
            current = model()
            path = Path(tmp)/'model.pt'
            save_checkpoint(path, current, current.make_optimizer(), 0)
            original = torch.load(path, weights_only=False)
            self.assertEqual(original['format'], CHECKPOINT_FORMAT)
            for version in ('smolgs-checkpoint-v3', 'smolgs-checkpoint-v4-explicit', 'smolgs-checkpoint-v5-ecvq'):
                payload = {**original, 'format': version}
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, 'Unsupported checkpoint format'):
                    load_checkpoint(path)
            archive_path = Path(tmp)/'old.zip'
            for version in ('smolgs-archive-v6', 'smolgs-archive-v7-explicit', 'smolgs-archive-v8-ecvq',
                            'smolgs-archive-v9-neural', 'smolgs-archive-v9-explicit', 'smolgs-archive-v9-ecvq'):
                with zipfile.ZipFile(archive_path, 'w') as archive:
                    archive.writestr('metadata.json', json.dumps({'format': version}))
                with self.assertRaisesRegex(ValueError, 'Unsupported archive format'):
                    decode_model(archive_path)

    def test_standalone_global_and_regional_ecvq_roundtrip(self):
        for depth, tail in ((18, 0.), (16, .1)):
            with self.subTest(depth=depth, tail=tail), tempfile.TemporaryDirectory() as tmp:
                current = model(coordinate_depth=depth, coordinate_tail_fraction=tail)
                optimizer = current.make_optimizer()
                current.prepare_geometry_quantization(25)
                original_bounds = current.bounds_min.clone(), current.bounds_max.clone()
                current.freeze_coordinates(optimizer)
                self.assertEqual(current.config.octree_depth, 16)
                for a, b in zip(original_bounds, (current.bounds_min, current.bounds_max)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                current.replace_topology(torch.tensor([3, 0, 2, 1]), optimizer)
                path = Path(tmp)/'model.zip'
                encode_model(current, path, 'cpu', chunk_splats=2)
                with zipfile.ZipFile(path) as archive:
                    self.assertEqual(json.loads(archive.read('metadata.json'))['format'], ARCHIVE_FORMAT)
                reference = export_reference(current)
                decoded, _ = decode_model(path)
                for name in ('xyz', 'cells', 'coordinate_cells', 'coordinate_regions'):
                    torch.testing.assert_close(getattr(reference, name), getattr(decoded, name), rtol=0, atol=0)
                camera = Camera('probe', 16, 16, 14., 14., torch.eye(4))
                a, b = render(reference, camera, 25, 'torch'), render(decoded, camera, 25, 'torch')
                torch.testing.assert_close(a['image'], b['image'], atol=2e-6, rtol=2e-5)
                for branch in (current, decoded):
                    checkpoint = Path(tmp)/('decoded.pt' if bool(branch.decoded) else 'training.pt')
                    save_checkpoint(checkpoint, branch, optimizer if branch is current else None, 25)
                    restored, _, _, _ = load_checkpoint(checkpoint, resume=branch is current)
                    torch.testing.assert_close(branch.xyz, restored.xyz, rtol=0, atol=0)
                    torch.testing.assert_close(branch.coordinate_cells, restored.coordinate_cells, rtol=0, atol=0)

    def test_scored_collision_and_atomic_rejection(self):
        current = model(coordinate_depth=18, coordinate_max_collision_fraction=.75)
        optimizer = current.make_optimizer()
        # Materialize Adam state to exercise migration to selected survivors.
        sum(getattr(current, n).square().sum() for n in current.point_parameter_names).backward()
        optimizer.step()
        with torch.no_grad():
            current.xyz[1].copy_(current.xyz[0])
        expected = current.features[1].detach().clone()
        removed = current.freeze_coordinates(optimizer, scores=torch.tensor([0., 5., 1., 1.]))
        self.assertEqual(removed, 1)
        self.assertTrue(bool((current.features == expected).all(-1).any()))
        self.assertEqual(len(current.coordinate_cells), 3)
        self.assertEqual(optimizer.state[current.xyz]['exp_avg'].shape, (3,3))
        current = model(coordinate_depth=18, coordinate_max_collision_fraction=0.)
        with torch.no_grad():
            current.xyz[1].copy_(current.xyz[0])
        before = copy.deepcopy(current.state_dict())
        with self.assertRaises(RuntimeError):
            current.freeze_coordinates()
        for name, value in before.items():
            torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=0)

    def test_empty_main_region_and_invalid_coordinate_metadata(self):
        current = model(coordinate_depth=16, coordinate_tail_fraction=.1)
        current.freeze_coordinates()
        files, layout, order, _ = encode_coordinates(current)
        self.assertEqual(layout['regions'][0]['count'], 0)
        bad = copy.deepcopy(layout)
        bad['regions'][1]['count'] += 1
        with self.assertRaises(ValueError):
            decode_coordinates(current, files, bad, len(order))

    def test_populated_main_and_exterior_native_codec(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as tmp:
                current = model(device, coordinate_depth=16, coordinate_tail_fraction=.001)
                inner = torch.linspace(-.2, .2, 18, device=device)
                points = torch.stack([inner, inner*.5, inner*.3+2], -1)
                points = torch.cat([points, points.new_tensor([[-100., -100., -100.], [100.,100.,100.]])])
                current.initialize(points)
                current.prepare_geometry_quantization(25)
                current.freeze_coordinates()
                counts = torch.bincount(current.coordinate_regions, minlength=2)
                self.assertEqual(counts.tolist(), [18, 2])
                path = Path(tmp)/'dual.zip'
                codec = 'cuda' if device == 'cuda' else 'cpu'
                encode_model(current, path, codec)
                reference = export_reference(current, codec)
                decoded, _ = decode_model(path, device)
                for name in ('xyz', 'cells', 'features', 'coordinate_cells', 'coordinate_regions',
                             'decoded_scales', 'scale_indices', 'rot_indices'):
                    torch.testing.assert_close(getattr(reference, name), getattr(decoded, name), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
