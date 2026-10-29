"""Opacity penalty is off for new runs; image gradients and pruning remain usable."""
from pathlib import Path
import unittest

import torch
from arguments import Config
from scene.dataset_readers import Camera
from scene.gaussian_model import SmolGS
from gaussian_renderer import render
from utils.loss_utils import loss_terms, ssim
from test_core import tiny_config


class NoOpacityRegularizationTests(unittest.TestCase):
    def test_regularizer_is_removed_and_old_fields_are_rejected(self):
        self.assertNotIn('lambda_opacity', Config().to_dict())
        for variant in ('small', 'base', 'large'):
            self.assertFalse(hasattr(Config.load(variant=variant), 'lambda_opacity'))
        root = Path(__file__).resolve().parents[1]
        for path in (root/'configs').glob('*.json'):
            with self.subTest(path=path.name):
                self.assertNotIn('lambda_opacity', Config.load(path).to_dict())
        saved = {**Config().to_dict(), 'lambda_opacity': 3e-7}
        with self.assertRaises(TypeError):
            Config.from_saved(saved)
        self.assertEqual(saved['lambda_opacity'], 3e-7)

    def test_compaction_image_loss_still_trains_opacity(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        model = SmolGS(tiny_config())
        model.initialize(torch.tensor([[-.1,0.,2.],[.1,0.,2.1]]))
        camera = Camera('probe',16,16,14.,14.,torch.eye(4))
        package = render(model,camera,9,'torch')
        target = torch.zeros_like(package['image'])
        loss, terms = loss_terms(model,package,target,9)
        expected = .8*(package['image']-target).abs().mean()+.2*(1-ssim(package['image'],target))
        torch.testing.assert_close(loss,expected,rtol=0,atol=0)
        self.assertNotIn('opacity', terms)
        params = [p for p in model.opacity_mlp.parameters() if p.requires_grad]
        actual = torch.autograd.grad(loss,params,retain_graph=True)
        photo = torch.autograd.grad(expected,params)
        self.assertGreater(sum(float(g.abs().sum()) for g in actual),0.)
        for a,b in zip(actual,photo):
            torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
