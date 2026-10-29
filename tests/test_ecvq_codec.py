"""rANS roundtrip, index remapping, standalone decode and rendering."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from gaussian_renderer import render
from scene.dataset_readers import Camera
from smolgs.codec import encode_model, decode_model, export_reference, _ecvq_geometry
from smolgs.vq.coding import cdf_from_logits, encode_indices, decode_indices
from test_ecvq import model
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.graphics_utils import quaternion_matrix


class ECVQCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_export_repairs_negative_scales_without_mutating_training_model(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                current = model(device)
                current.prepare_geometry_quantization(25)
                current.freeze_coordinates()
                with torch.no_grad():
                    current.quantizers['scale'].codebook.copy_(current.xyz.new_tensor(
                        [[[-.2]*3, [1e-10]*3, [0.]*3, [.3]*3]]))
                    current.log_scales.copy_(current.xyz.new_tensor(
                        [[1e-8]*3, [1e-10]*3, [1e-12]*3, [.3]*3]).log())
                    current.quantizers['scale'].logits.zero_()
                before = copy.deepcopy(current.state_dict())
                reference = copy.deepcopy(current.quantizers['scale'])
                with torch.no_grad():
                    reference.codebook[reference.codebook < 0] = 1e-8
                    assigned = reference(current.log_scales.exp(), True)
                used, index = np.unique(assigned['x_index'].cpu().numpy(), return_inverse=True)
                actual = _ecvq_geometry(current, torch.arange(4, device=device))['scale']
                np.testing.assert_array_equal(actual['indices'], index[:, None])
                np.testing.assert_array_equal(actual['codebook'], reference.codebook[0, used].detach().cpu().numpy())
                self.assertEqual(len(used), 4)  # Negative, tiny positive and zero are distinguished.
                path = Path(directory)/'scales.zip'
                encode_model(current, path, 'cpu')
                decoded, _ = decode_model(path, device)
                expected = export_reference(current)
                torch.testing.assert_close(decoded.decoded_scales, expected.decoded_scales, rtol=0, atol=0)
                self.assertTrue((decoded.decoded_scales == 0).any())
                checkpoint = Path(directory)/'decoded.pt'
                save_checkpoint(checkpoint, decoded, None, 25)
                restored, _, _, _ = load_checkpoint(checkpoint, device)
                torch.testing.assert_close(restored.decoded_scales, decoded.decoded_scales, rtol=0, atol=0)
                for name, value in before.items():
                    torch.testing.assert_close(current.state_dict()[name], value, rtol=0, atol=0)

    def test_cdf_index_roundtrip_and_single_symbol(self):
        for codes in (1, 2, 17, 8192):
            logits = torch.linspace(-10., 1., codes).reshape(1, -1)
            cdf = cdf_from_logits(logits)
            self.assertEqual(cdf[0], 0)
            self.assertEqual(cdf[-1], 65536)
            self.assertTrue((np.diff(cdf) > 0).all())
            for count in (1, 31):
                indexes = np.arange(count, dtype=np.int64)[:, None] % codes
                stream = encode_indices(indexes, cdf)
                np.testing.assert_array_equal(decode_indices(stream, cdf, count), indexes)

    def test_shrink_remap_preserves_selected_values_and_single_point(self):
        current = model(vq_scale_cbsize=8, vq_rot_cbsize=8)
        current.prepare_geometry_quantization(17)
        order = torch.tensor([3, 1, 0, 2])
        actual = _ecvq_geometry(current, order)
        for key, values in (('scale', current.log_scales[order].exp()), ('rot', current.rotations[order])):
            quantizer = current.quantizers[key]
            expected = quantizer(values, True)['x_hat'].detach().numpy()
            result = actual[key]
            np.testing.assert_array_equal(result['codebook'][result['indices'][:, 0]], expected)
            self.assertEqual(len(np.unique(result['indices'])), len(result['codebook']))
        current.replace_topology(torch.tensor([0]))
        current.freeze_coordinates()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'one.zip'
            encode_model(current, path)
            restored, _ = decode_model(path)
            self.assertEqual(restored.scale_indices.shape, (1, 1))
            self.assertEqual(restored.quantizers['scale'].codebook.shape, (1, 1, 3))

    def test_standalone_geometry_covariance_render_and_decoded_checkpoint(self):
        options = [('cpu', False)]+([('cuda', False), ('cuda', True)] if torch.cuda.is_available() else [])
        for device, native_networks in options:
            with self.subTest(device=device, native=native_networks), tempfile.TemporaryDirectory() as directory:
                current = model(device, vq_scale_cbsize=8, vq_rot_cbsize=8,
                                appearance_backend='tcnn' if native_networks else 'torch',
                                octree_backend='tcnn' if native_networks else 'torch',
                                octree_mlp_width=64 if native_networks else 8,
                                hash_backend='cuda' if native_networks else 'torch')
                current.prepare_geometry_quantization(25)
                current.freeze_coordinates()
                current.replace_topology(torch.tensor([3, 0, 2, 1], device=device))
                before = copy.deepcopy(current.state_dict())
                for codec in ['cpu']+(['cuda'] if device == 'cuda' else []):
                    path = Path(directory)/f'{codec}.zip'
                    report = encode_model(current, path, codec, chunk_splats=1)
                    self.assertEqual(report['geometry_storage'], 'rdo-rans16-v1')
                    reference = export_reference(current, codec)
                    decoded, _ = decode_model(path, device)
                    with zipfile.ZipFile(path) as archive:
                        self.assertNotIn('geometry/log_scales.npy', archive.namelist())
                        self.assertNotIn('geometry/rotations.npy', archive.namelist())
                    for key in ('scale', 'rot'):
                        torch.testing.assert_close(getattr(decoded, key+'_indices'), getattr(reference, key+'_indices'), rtol=0, atol=0)
                        torch.testing.assert_close(decoded.quantizers[key].codebook, reference.quantizers[key].codebook, rtol=0, atol=0)
                    self.assertEqual(path.stat().st_size, report['bytes'])
                    for shift in (0., .1):
                        transform = torch.eye(4, device=device)
                        transform[0, 3] = shift
                        camera = Camera('probe', 16, 16, 14., 14., transform)
                        backend = 'cuda' if device == 'cuda' else 'torch'
                        a = render(reference, camera, 25, backend)
                        b = render(decoded, camera, 25, backend)
                        for name in ('xyz', 'scales', 'rotation', 'color', 'opacity'):
                            torch.testing.assert_close(a['attributes'][name], b['attributes'][name], rtol=2e-5, atol=2e-6)
                        for package in (a, b):
                            attrs = package['attributes']
                            factor = quaternion_matrix(attrs['rotation'])*attrs['scales'][:, None, :]
                            package['covariance'] = factor @ factor.transpose(1, 2)
                        torch.testing.assert_close(a['covariance'], b['covariance'], rtol=0, atol=0)
                        torch.testing.assert_close(a['image'], b['image'], rtol=2e-5, atol=2e-6)
                    checkpoint = Path(directory)/'decoded.pt'
                    save_checkpoint(checkpoint, decoded, None, 25)
                    restored, _, _, _ = load_checkpoint(checkpoint, device)
                    torch.testing.assert_close(restored.decoded_scales, decoded.decoded_scales, rtol=0, atol=0)
                    torch.testing.assert_close(render(restored, camera, 25, backend)['image'], b['image'], rtol=0, atol=0)
                    with self.assertRaisesRegex(ValueError, 'inference-only'):
                        load_checkpoint(checkpoint, device, resume=True)
                for name, value in before.items():
                    torch.testing.assert_close(current.state_dict()[name], value, rtol=0, atol=0)

    def test_corrupt_cdf_codebook_and_checksums_are_rejected(self):
        current = model()
        current.prepare_geometry_quantization(25)
        current.freeze_coordinates()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.zip'
            encode_model(current, path)
            with zipfile.ZipFile(path) as archive:
                original = {name: archive.read(name) for name in archive.namelist()}
            for corruption in ('checksum', 'cdf', 'codebook'):
                files = dict(original)
                meta = json.loads(files['metadata.json'])
                name = meta['geometry']['scale']['cdf' if corruption != 'codebook' else 'codebook']
                value = np.load(io.BytesIO(files[name]))
                if corruption == 'codebook': value[0, 0] = -1.
                else: value[-1] = 0
                stream = io.BytesIO(); np.save(stream, value, allow_pickle=False)
                files[name] = stream.getvalue()
                if corruption != 'checksum': meta['files_sha256'][name] = hashlib.sha256(files[name]).hexdigest()
                files['metadata.json'] = json.dumps(meta).encode()
                with zipfile.ZipFile(path, 'w') as archive:
                    for key, data in files.items(): archive.writestr(key, data)
                with self.assertRaises(ValueError): decode_model(path)


if __name__ == '__main__':
    unittest.main()
