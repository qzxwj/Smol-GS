"""Regression checks for Inria LPIPS scores and protocol provenance."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from smolgs.evaluation import evaluate
from utils.loss_utils import LPIPS

ROOT = Path(__file__).resolve().parents[1]


class InriaLPIPSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        weights = Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
        if not weights.is_file():
            raise unittest.SkipTest('Cached VGG16 weights required; tests never download weights')
        torch.set_num_threads(2)

    def test_inria_reference_score_and_evaluation_entry_points(self):
        # Golden score independently computed with Inria lpipsPyTorch VGG v0.1.
        # It detects applying 2*x-1 even when all callers agree with each other.
        image = (torch.arange(3*64*64).reshape(3, 64, 64) % 251).float()/250
        target = (image.roll((7, 11), (-2, -1))*.7+.1).clamp(0, 1)
        metric = LPIPS('cpu')
        self.assertAlmostEqual(float(metric(image, target)), 0.32033008337020874, delta=1e-6)
        self.assertAlmostEqual(float(metric(image, image)), 0., delta=1e-7)
        model = torch.nn.Module()
        model.xyz = torch.nn.Parameter(torch.zeros(2, 3))
        camera = SimpleNamespace(name='fixture', to=lambda device: camera,
                                 image=lambda device: target)
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch('smolgs.evaluation.LPIPS', return_value=metric,
                            protocol=LPIPS.protocol, status=LPIPS.status), mock.patch(
                    'smolgs.evaluation.render', return_value={'image': image}):
                report = evaluate(model, [camera], 7000, 'torch', folder)
            self.assertTrue(model.training)
            self.assertAlmostEqual(report['lpips'], 0.32033008337020874, delta=1e-6)
            self.assertEqual(report['lpips_protocol'], LPIPS.protocol)
            self.assertEqual(json.loads((Path(folder)/'metrics.json').read_text())['lpips_protocol'], LPIPS.protocol)
            # Exercise the actual offline CLI on saved PNGs in a new process.
            method = Path(folder)/'test/ours_7000'
            for name, value in [('renders', image), ('gt', target)]:
                (method/name).mkdir(parents=True)
                pixels = (value.permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
                Image.fromarray(pixels).save(method/name/'fixture.png')
            def read(path):
                with Image.open(path) as im:
                    return torch.from_numpy(np.array(im, dtype=np.float32)/255).permute(2, 0, 1)
            expected = float(metric(read(method/'renders/fixture.png'), read(method/'gt/fixture.png')))
            subprocess.run([sys.executable, str(ROOT/'metrics.py'), '-m', folder, '--device', 'cpu'],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            result = json.loads((Path(folder)/'results.json').read_text())
            self.assertAlmostEqual(result['ours_7000']['LPIPS'], expected, delta=1e-6)
            protocol = json.loads((Path(folder)/'evaluation_protocol.json').read_text())
            self.assertEqual(protocol['lpips_protocol'], LPIPS.protocol)

    def test_collector_does_not_mix_old_and_inria_lpips(self):
        with tempfile.TemporaryDirectory() as folder:
            for scene, protocol in [('truck', LPIPS.protocol), ('train', 'VGG, normalize inputs to [-1,1]')]:
                path = Path(folder)/'tandt'/scene/'base/test/ours_35000/metrics.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'psnr': 25., 'ssim': .8, 'lpips': .2,
                                            'fps': 50., 'splats': 10, 'views': 32,
                                            'lpips_status': protocol}))
            subprocess.run([sys.executable, str(ROOT/'scripts/collect_results.py'), '--output-root', folder],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            report = json.loads((Path(folder)/'benchmark_results.json').read_text())
            self.assertIsNone(report['dataset_averages'][0]['lpips'])
            self.assertEqual(report['dataset_averages'][0]['psnr'], 25.)
            self.assertEqual(len(report['dataset_averages'][0]['lpips_protocols']), 2)


if __name__ == '__main__':
    unittest.main()
