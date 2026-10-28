"""CPU arithmetic coding must preserve the model's FP32 integer decisions."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from arguments import Config
from scene.gaussian_model import SmolGS
from smolgs.codec import encode_model, decode_model
from utils.arithmetic import (encode_gaussian, encode_gaussian_symbols,
                              decode_gaussian_symbols)


class CpuSymbolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @torch.no_grad()
    def assert_archive_symbols(self, path, model):
        """Compare stored integers directly; no symbol digests in the format."""
        mf, sf, ms, ss, qf, qs = model.prior(model.xyz)
        attributes = [('features', model.features, mf, sf, qf),
                      ('scales', model.log_scales.exp(), ms, ss, qs)]
        with zipfile.ZipFile(path) as archive:
            meta = json.loads(archive.read('metadata.json'))
            for chunk in meta['chunks']:
                start, end = chunk['start'], chunk['end']
                for label, values, means, stds, steps in attributes:
                    self.assertNotIn(label+'_sha256', chunk)
                    expected = torch.round(values[start:end]/steps[start:end]).to(torch.int64).numpy().reshape(-1)
                    actual = np.empty_like(expected)
                    contexts = [a[start:end].expand_as(values[start:end]).numpy().reshape(-1)
                                for a in (means, stds, steps)]
                    description = chunk[label]
                    parts = ([{'start': 0, 'end': expected.size, 'file': description}]
                             if isinstance(description, str) else description)
                    for part in parts:
                        first, last = part['start'], part['end']
                        actual[first:last] = decode_gaussian_symbols(
                            archive.read(part['file']), *[a[first:last] for a in contexts])
                    np.testing.assert_array_equal(actual, expected)

    def test_integer_codec_preserves_symbols_without_float_reconstruction(self):
        rng = np.random.default_rng(19)
        cases = [rng.integers(-1000, 1001, 257, dtype=np.int64),
                 np.full(5, 2**24+1, dtype=np.int64),
                 np.full(5, -(2**24+1), dtype=np.int64)]
        for expected in cases:
            with self.subTest(first_symbol=int(expected[0])):
                means = np.zeros(len(expected), dtype=np.float32)
                stds = np.ones_like(means)
                steps = np.full_like(means, 0.3)
                blob = encode_gaussian_symbols(expected, means, stds, steps)
                actual = decode_gaussian_symbols(blob, means, stds, steps)
                self.assertEqual(actual.dtype, np.int64)
                np.testing.assert_array_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, 'integers'):
            encode_gaussian_symbols(np.array([1.5]), [0.], [1.], [1.])

    def test_float_wrapper_keeps_input_quantization_precision(self):
        q = np.nextafter(np.float32(1.), np.float32(2.))
        v = np.nextafter(np.float32(1.5), np.float32(2.))
        values = np.array([v, -v], dtype=np.float32)
        steps = np.full_like(values, q)
        blob = encode_gaussian(values, np.zeros(2), np.ones(2), steps)
        np.testing.assert_array_equal(decode_gaussian_symbols(blob, np.zeros(2), np.ones(2), steps),
                                      np.array([2, -2], dtype=np.int64))

    def test_legacy_gac1_bitstream_remains_compatible(self):
        # Produced by the original float-valued encoder for [-3, 0, 2],
        # with mean=0, std=1, step=1. The integer API keeps this wire format.
        legacy = bytes.fromhex('474143310300000000000000fdffffffffffffff02000000000000000110')
        symbols = np.array([-3, 0, 2], dtype=np.int64)
        means, stds, steps = np.zeros(3), np.ones(3), np.ones(3)
        np.testing.assert_array_equal(decode_gaussian_symbols(legacy, means, stds, steps), symbols)
        self.assertEqual(encode_gaussian_symbols(symbols, means, stds, steps), legacy)

    def test_archive_preserves_fp32_rounding_at_half_integer_boundaries(self):
        # The ratio rounds to 1.5 in FP32, but lies below 1.5 in FP64.
        q = torch.nextafter(torch.tensor(1.), torch.tensor(2.))
        value = torch.nextafter(torch.tensor(1.5), torch.tensor(2.))
        self.assertEqual(int(torch.round(value/q)), 2)
        self.assertEqual(int(np.rint(float(value)/float(q))), 1)
        positive = torch.stack([torch.nextafter(value, torch.tensor(-float('inf'))),
                                value, torch.nextafter(value, torch.tensor(float('inf')))])
        boundary_values = torch.cat([positive, -positive, torch.tensor([0., 0.5])])
        config = Config(hash_backend='torch', mlp_width=16, octree_mlp_width=8,
                        resolutions_3d=(4,), resolutions_2d=(4,),
                        hash_log2_3d=5, hash_log2_2d=5, scale_step=1.)
        model = SmolGS(config)
        model.initialize(torch.tensor([[0., 0., 2.], [.1, .2, 2.1], [.3, .1, 2.2]]))
        with torch.no_grad():
            for parameter in model.entropy_mlp.parameters():
                parameter.zero_()
            # q_f = 1 + tanh(bias), q_s = scale_step * (1 + tanh(bias)).
            model.entropy_mlp[4].bias[-2:] = q-1
            model.features.copy_(boundary_values.expand_as(model.features))
            model.log_scales.copy_(positive.log().expand_as(model.log_scales))
            model.freeze_coordinates()
        model.eval()
        with torch.no_grad():
            _, _, _, _, qf, qs = model.prior(model.xyz)
            expected_symbols = torch.round(model.features/qf).to(torch.int64)
            expected_scale_symbols = torch.round(model.log_scales.exp()/qs).to(torch.int64)
            expected_features = expected_symbols.float()*qf
            expected_scales = (expected_scale_symbols.float()*qs).clamp_min(1e-7)
        for chunk_size in (1, 2, len(model.xyz)):
            with self.subTest(chunk_size=chunk_size), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'boundary.zip'
                encode_model(model, path, 'cpu', chunk_splats=chunk_size)
                restored, _ = decode_model(path)
                torch.testing.assert_close(restored.features, expected_features, atol=0, rtol=0)
                torch.testing.assert_close(restored.log_scales.exp(), expected_scales,
                                           atol=1e-7, rtol=1e-6)
                self.assert_archive_symbols(path, model)

        # Force both recursive splat splitting and single-splat attribute
        # segmentation while retaining the rounding-boundary values above.
        with torch.no_grad():
            model.features[:, -1] = 2_000_000.
            expected_features = torch.round(model.features/qf)*qf
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'segmented.zip'
            report = encode_model(model, path, 'cpu', chunk_splats=len(model.xyz))
            self.assertGreater(report['segmented_attributes'], 0)
            restored, _ = decode_model(path)
            torch.testing.assert_close(restored.features, expected_features, atol=0, rtol=0)
            torch.testing.assert_close(restored.log_scales.exp(), expected_scales, atol=1e-7, rtol=1e-6)

            self.assert_archive_symbols(path, model)


if __name__ == '__main__':
    unittest.main()
