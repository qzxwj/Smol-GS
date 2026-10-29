import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from gaussian_renderer import render
from scene.dataset_readers import Camera
from smolgs.coordinate_selection import score_training_opacity, select_coordinate_survivors
from test_ecvq import model, config
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.graphics_utils import quantize_cells


class CoordinateSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_best_per_cell_ties_and_invalid_scores(self):
        codes = torch.tensor([9, 3, 9, 3, 3, 12])
        scores = torch.tensor([.2, .7, .9, .7, .1, 0.], dtype=torch.float64)
        self.assertEqual(select_coordinate_survivors(codes, scores).tolist(), [1, 2, 5])
        self.assertEqual(select_coordinate_survivors(codes).tolist(), [1, 0, 5])
        self.assertEqual(select_coordinate_survivors(codes, scores*0).tolist(), [1, 0, 5])
        for bad in (scores[:-1], scores.long(), scores-1, scores*float('nan')):
            with self.assertRaises(ValueError):
                select_coordinate_survivors(codes, bad)

    def test_scores_include_zero_for_invisible_views_and_restore_state(self):
        current = model()
        current.prepare_geometry_quantization(13)
        current.color_mlp.eval()  # Mixed module modes must be preserved too.
        cameras = [Camera(str(i), 8, 8, 8., 8., torch.eye(4)) for i in range(2)]
        opacities = [torch.tensor([.9, .8, .7, .1]), torch.tensor([.1, .6, .5, .9])]
        visibility = [torch.tensor([True, False, True, False]), torch.tensor([True, True, False, False])]
        before = copy.deepcopy(current.state_dict())
        modes = [m.training for m in current.modules()]
        random.seed(81); np.random.seed(81); torch.manual_seed(81)
        expected_rng = (random.random(), np.random.rand(), torch.rand(3))
        random.seed(81); np.random.seed(81); torch.manual_seed(81)

        def fake_render(candidate, camera, step, backend):
            self.assertFalse(candidate.training)
            random.random(); np.random.rand(); torch.rand(3)
            candidate.scale_index_cache.fill_(3)
            i = int(camera.name)
            return {'attributes': {'opacity': opacities[i][:, None]}, 'visible': visibility[i]}

        with mock.patch('gaussian_renderer.render', side_effect=fake_render), \
                mock.patch.object(Camera, 'image', side_effect=AssertionError('Scoring must not read GT')):
            scores, seen = score_training_opacity(current, cameras, 13, 'torch', progress=False)
        torch.testing.assert_close(scores, torch.tensor([.5, .3, .35, 0.], dtype=torch.float64))
        self.assertEqual(seen.tolist(), [2, 1, 1, 0])
        self.assertEqual([m.training for m in current.modules()], modes)
        for name, value in before.items():
            torch.testing.assert_close(current.state_dict()[name], value, rtol=0, atol=0)
        actual_rng = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(actual_rng[:2], expected_rng[:2])
        torch.testing.assert_close(actual_rng[2], expected_rng[2], rtol=0, atol=0)
        with mock.patch('gaussian_renderer.render', side_effect=RuntimeError('probe failure')):
            with self.assertRaisesRegex(RuntimeError, 'probe failure'):
                score_training_opacity(current, cameras, 13, progress=False)
        self.assertEqual([m.training for m in current.modules()], modes)
        for cameras_bad in ([], cameras+ cameras[:1]):
            with self.assertRaises(ValueError):
                score_training_opacity(current, cameras_bad, 13, progress=False)

    def test_real_renderer_scores_cpu_cuda_without_gt_or_state_changes(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                current = model(device)
                current.prepare_geometry_quantization(13)
                cameras = [Camera('a', 16, 16, 14., 14., torch.eye(4)),
                           Camera('b', 16, 16, 14., 14., torch.eye(4))]
                cameras[1].world_to_camera[0, 3] = .2
                backend = 'cuda' if device == 'cuda' else 'torch'
                current.eval()
                expected = torch.zeros(4, dtype=torch.float64, device=device)
                with torch.no_grad():
                    for camera in cameras:
                        package = render(current, camera.to(device), 13, backend)
                        expected += torch.where(package['visible'], package['attributes']['opacity'][:, 0].double(), 0.)/2
                current.train()
                before = copy.deepcopy(current.state_dict())
                rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else torch.get_rng_state().clone()
                with mock.patch.object(Camera, 'image', side_effect=AssertionError('GT not allowed')):
                    scores, _ = score_training_opacity(current, cameras, 13, backend, progress=False)
                torch.testing.assert_close(scores, expected, rtol=0, atol=0)
                for name, value in before.items():
                    torch.testing.assert_close(current.state_dict()[name], value, rtol=0, atol=0)
                torch.testing.assert_close(torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_scored_freeze_migrates_parameters_adam_caches_and_checkpoint(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                current = model(device, coordinate_max_collision_fraction=.75)
                optimizer = current.make_optimizer()
                current.prepare_geometry_quantization(13)
                sum(getattr(current, n).square().sum() for n in current.point_parameter_names).backward()
                optimizer.step()
                with torch.no_grad():
                    current.xyz.copy_(current.xyz.new_tensor([[0., 0., 2.], [0., 0., 2.], [1., 0., 2.], [1., 0., 2.]]))
                    current.scale_index_cache.copy_(torch.arange(4, device=device)[:, None])
                    current.rot_index_cache.copy_(torch.arange(4, device=device).flip(0)[:, None])
                before = copy.deepcopy(current.state_dict())
                moments = {n: copy.deepcopy(optimizer.state[getattr(current, n)]) for n in current.point_parameter_names}
                _, centers = quantize_cells(current.xyz, current.xyz.amin(0), torch.maximum(current.xyz.amax(0), current.xyz.amin(0)+1e-6), current.config.octree_depth)
                scores = current.xyz.new_tensor([.1, .9, .5, .5])
                self.assertEqual(current.freeze_coordinates(optimizer, scores=scores), 2)
                indices = torch.tensor([1, 2], device=device)
                for name in current.point_parameter_names:
                    torch.testing.assert_close(getattr(current, name), centers[indices] if name == 'xyz' else before[name][indices], rtol=0, atol=0)
                    for key in ('exp_avg', 'exp_avg_sq'):
                        torch.testing.assert_close(optimizer.state[getattr(current, name)][key], moments[name][key][indices], rtol=0, atol=0)
                for key in ('scale', 'rot'):
                    torch.testing.assert_close(getattr(current, key+'_index_cache'), before[key+'_index_cache'][indices])
                path = Path(directory)/'frozen.pt'
                save_checkpoint(path, current, optimizer, 13, {'coordinate_selection': 'opacity'})
                restored, _, _, meta = load_checkpoint(path, device, resume=True)
                self.assertEqual(meta['coordinate_selection'], 'opacity')
                for name, value in current.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    def test_training_entrypoint_scores_training_views_only_and_resumes_policy(self):
        from train import main
        cameras = [Camera('train_a', 16, 16, 14., 14., torch.eye(4)),
                   Camera('train_b', 16, 16, 14., 14., torch.eye(4))]
        cameras[1].world_to_camera[0, 3] = .2
        test = [Camera('test_only', 16, 16, 14., 14., torch.eye(4))]
        points = model().xyz.detach().numpy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = root/'config.json'
            cfg.write_text(json.dumps(config(densify_threshold=1e6).to_dict()))
            args = ['-s', str(root/'data'), '-m', str(root/'run'), '--config', str(cfg),
                    '--device', 'cpu', '--renderer', 'torch', '--skip-export', '--test-iterations',
                    '--checkpoint-interval', '25', '--log-interval', '25']
            with mock.patch('train.load_scene', return_value=(cameras, test)), \
                    mock.patch('train.load_points', return_value=points), \
                    mock.patch.object(Camera, 'image', return_value=torch.full((3, 16, 16), .25)), \
                    mock.patch('train.score_training_opacity', wraps=score_training_opacity) as score:
                main([*args, '--coordinate-selection', 'opacity', '--stop-after', '19'])
                self.assertEqual(score.call_count, 1)
                self.assertIs(score.call_args.args[1], cameras)
                receipt = torch.load(root/'run/coordinate_opacity_scores.pt', weights_only=False)
                self.assertEqual(receipt['train_names'], ['train_a', 'train_b'])
                latest = str(root/'run/checkpoint_latest.pt')
                with self.assertRaisesRegex(ValueError, 'Cannot change coordinate selection'):
                    main([*args, '--resume', latest, '--coordinate-selection', 'first'])
                main([*args, '--resume', latest])
                self.assertEqual(score.call_count, 1)
                run = json.loads((root/'run/run.json').read_text())
                self.assertEqual(run['coordinate_selection'], 'opacity')


if __name__ == '__main__':
    unittest.main()
