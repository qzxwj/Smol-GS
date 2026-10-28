"""Paper opacity and native appearance networks, gradients, and codec roundtrips."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch
from arguments import Config, config_from_args
from scene.gaussian_model import SmolGS
from scene.dataset_readers import Camera
from gaussian_renderer import render
from smolgs.codec import encode_model, decode_model, quantized_hash_reference
from train import parser
from utils.checkpoint import save_checkpoint, load_checkpoint


def config(**updates):
    values = dict(hash_backend='torch', resolutions_3d=(4, 6), resolutions_2d=(4,), hash_features=2, hash_log2_2d=5,
                  hash_log2_3d=5,
                  mlp_width=16, octree_mlp_width=8)
    values.update(updates)
    return Config(**values)


class AppearanceConfigTests(unittest.TestCase):
    def test_only_paper_opacity_is_supported(self):
        self.assertEqual(config().opacity_activation, 'abs_tanh')
        with self.assertRaisesRegex(ValueError, 'opacity_activation'):
            config(opacity_activation='sigmoid')
        with self.assertRaisesRegex(ValueError, 'must record paper'):
            Config.from_saved({'appearance_backend': 'tcnn'})
        recorded = {'appearance_backend': 'tcnn', 'opacity_activation': 'abs_tanh'}
        self.assertEqual(Config.from_saved(recorded).opacity_activation, 'abs_tanh')

    def test_native_backend_cli_and_device_validation(self):
        args = parser().parse_args(['-s', '/tmp/data', '-m', '/tmp/out',
                                   '--appearance-backend', 'tcnn'])
        self.assertEqual(config_from_args(args).appearance_backend, 'tcnn')
        with self.assertRaisesRegex(ValueError, 'requires device=cuda'):
            SmolGS(config(appearance_backend='tcnn'))


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('tinycudann'), 'Native CUDA environment required')
class NativeAppearanceTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(22)
        model = SmolGS(config(appearance_backend='tcnn'), device='cuda')
        points = torch.randn(12, 3, device='cuda')*0.12
        points[:, 2] += 2
        model.initialize(points)
        return model

    def test_other_branches_and_rng_unchanged_no_bias_and_output_formula(self):
        torch.manual_seed(22)
        ordinary = SmolGS(config(), device='cuda')
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        torch.manual_seed(22)
        native = SmolGS(config(appearance_backend='tcnn'), device='cuda')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
        for name, value in ordinary.state_dict().items():
            if not name.startswith(('color_mlp.', 'opacity_mlp.')):
                torch.testing.assert_close(native.state_dict()[name], value, atol=0, rtol=0)
        self.assertEqual(list(dict(native.color_mlp.named_parameters())), ['network.params'])
        self.assertEqual(list(dict(native.opacity_mlp.named_parameters())), ['network.params'])
        native.initialize(torch.tensor([[0., 0., 2.], [0.1, 0., 2.]], device='cuda'))
        with torch.no_grad():
            native.color_mlp.network.params.zero_()
            native.opacity_mlp.network.params.zero_()
        attrs = native.attributes(torch.zeros(3, device='cuda'), 1000)
        torch.testing.assert_close(attrs['color'], torch.full((2, 3), 0.5, device='cuda'))
        torch.testing.assert_close(attrs['opacity'], torch.zeros((2, 1), device='cuda'))

    def test_paper_opacity_sign_symmetry_and_gradient_for_both_backends(self):
        class FixedRaw(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.raw = torch.nn.Parameter(torch.tensor([[0.], [-0.5], [0.5], [-2.], [2.]], device=device))

            def forward(self, inputs):
                return self.raw

        for backend, device in [('torch', 'cpu'), ('tcnn', 'cuda')]:
            with self.subTest(backend=backend):
                model = SmolGS(config(appearance_backend=backend), device=device)
                model.initialize(torch.tensor([[0., 0., 2.], [0.1, 0., 2.], [0., .1, 2.],
                                               [.1, .1, 2.], [.2, .1, 2.]], device=device))
                head = FixedRaw(device)
                model.opacity_mlp = head if backend == 'tcnn' else torch.nn.Sequential(head, torch.nn.Tanh())
                opacity = model.attributes(torch.zeros(3, device=device), 1000)['opacity']
                expected = torch.tensor([[0.], [.46211716], [.46211716], [.96402758], [.96402758]], device=device)
                torch.testing.assert_close(opacity, expected)
                opacity.sum().backward()
                self.assertTrue(torch.isfinite(head.raw.grad).all())
                self.assertEqual(float(head.raw.grad[0]), 0.)
                self.assertLess(float(head.raw.grad[1]), 0.)
                self.assertGreater(float(head.raw.grad[2]), 0.)
                torch.testing.assert_close(head.raw.grad[1], -head.raw.grad[2])


    def test_input_gradients_checkpoint_resume_and_archive(self):
        model = self.model()
        model.attributes(torch.zeros(3, device='cuda'), 1000)['opacity'].sum().backward()
        self.assertGreater(float(model.opacity_mlp.network.params.grad.norm()), 0.)
        self.assertGreater(float(model.features.grad.norm()), 0.)
        self.assertTrue(torch.isfinite(model.opacity_mlp.network.params.grad).all())
        model.zero_grad(set_to_none=True)
        inputs = torch.randn(17, 20, device='cuda', requires_grad=True)
        loss = model.color_mlp(inputs).square().mean()+model.opacity_mlp(inputs).square().mean()
        loss.backward()
        self.assertGreater(float(inputs.grad.norm()), 0)
        for module in (model.color_mlp, model.opacity_mlp):
            self.assertTrue(bool(torch.isfinite(module.network.params.grad).all()))
            self.assertGreater(float(module.network.params.grad.norm()), 0)
        optimizer = model.make_optimizer()
        model.update_learning_rates(optimizer, 1000)
        optimizer.step()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)/'checkpoint.pt'
            save_checkpoint(checkpoint, model, optimizer, 1000)
            restored, opt, step, _ = load_checkpoint(checkpoint, 'cuda', resume=True)
            self.assertEqual(step, 1000)
            self.assertEqual(restored.config.opacity_activation, 'abs_tanh')
            self.assertEqual(len(optimizer.state), len(opt.state))
            for current, current_opt in ((model, optimizer), (restored, opt)):
                current_opt.zero_grad(set_to_none=True)
                current.update_learning_rates(current_opt, 1001)
                current.color_mlp(inputs).square().mean().backward()
                current_opt.step()
            torch.testing.assert_close(model.color_mlp.network.params,
                                       restored.color_mlp.network.params, atol=1e-7, rtol=1e-5)
            model.freeze_coordinates()
            model.eval()
            reference = quantized_hash_reference(model)
            archive = Path(temporary)/'model.zip'
            encode_model(model, archive, 'cpu', chunk_splats=3)
            decoded, _ = decode_model(archive, 'cuda')
            self.assertEqual(decoded.config.opacity_activation, 'abs_tanh')
            camera = Camera('probe', 16, 16, 12., 12., torch.eye(4, device='cuda'))
            torch.testing.assert_close(render(reference, camera, 35000)['image'],
                                       render(decoded, camera, 35000)['image'], atol=1e-6, rtol=1e-6)
            with self.assertRaisesRegex(ValueError, 'requires --device cuda'):
                decode_model(archive, 'cpu')


if __name__ == '__main__':
    unittest.main()
