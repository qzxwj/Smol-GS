"""Native Morton-position MLP, gradients, checkpoints, and standalone decoding."""
from pathlib import Path
import tempfile
import unittest
import torch
from torch.nn import functional as F
from test_native_appearance import config
from scene.gaussian_model import SmolGS
from scene.dataset_readers import Camera
from gaussian_renderer import render
from smolgs.codec import encode_model, decode_model, quantized_hash_reference
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.graphics_utils import positional_bits


def native_config(**changes):
    values = dict(appearance_backend='tcnn', octree_backend='tcnn',
                  octree_init_seed=1337, octree_mlp_width=64)
    values.update(changes)
    return config(**values)


class PositionConfigTests(unittest.TestCase):
    def test_native_position_width_and_preset(self):
        with self.assertRaisesRegex(ValueError, 'FullyFusedMLP width'):
            native_config(octree_mlp_width=8)
        from arguments import Config
        current = Config.load(Path(__file__).resolve().parents[1]/'configs/native.json')
        self.assertEqual(current.learning_rates['octree_mlp'], [0.001, 0.001])
        self.assertEqual(current.opacity_activation, 'abs_tanh')
        self.assertEqual((current.appearance_backend, current.octree_backend), ('tcnn', 'tcnn'))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativePositionTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(22)
        model = SmolGS(native_config(), device='cuda')
        points = torch.randn(12, 3, device='cuda')*0.12
        points[:, 2] += 2
        model.initialize(points)
        return model

    def test_other_branches_rng_and_native_hidden_probe(self):
        torch.manual_seed(22)
        ordinary = SmolGS(config(appearance_backend='tcnn', octree_mlp_width=64), device='cuda')
        rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        torch.manual_seed(22)
        native = SmolGS(native_config(), device='cuda')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
        for name, value in ordinary.state_dict().items():
            if not name.startswith('octree_mlp.'):
                torch.testing.assert_close(native.state_dict()[name], value, atol=0, rtol=0)
        self.assertEqual(list(dict(native.octree_mlp.named_parameters())), ['network.params'])
        self.assertEqual(native.octree_mlp.network.params.numel(), 8192)
        cells = torch.randint(0, 65536, (113, 3), device='cuda')
        bits = positional_bits(cells, 16)
        _, _, hidden, output = native.octree_mlp.activity(bits)
        last = native.octree_mlp.network.params.detach().half()[-16*64:].reshape(16, 64)
        reconstructed = F.linear(hidden, last)[:, :8].float()
        torch.testing.assert_close(output, reconstructed, atol=0.003, rtol=0.003)
        native.position_embedding(cells).square().mean().backward()
        self.assertGreater(native.octree_mlp.hidden_gradient_norm(), 0)
        self.assertTrue(bool(torch.isfinite(native.octree_mlp.network.params.grad).all()))
        native.initialize(torch.tensor([[0., 0., 2.], [0.1, 0., 2.]], device='cuda'))
        with torch.no_grad(): native.octree_mlp.network.params.zero_()
        health = native.octree_health()
        self.assertEqual(health['octree_activity_method'], 'torch-fp16-reconstruction')
        self.assertEqual(health['octree_hidden_nonzero_point_fraction'], 0)
        self.assertEqual(health['octree_embedding_range'], 0)

    def test_checkpoint_resume_native_position_and_codec(self):
        model = self.model()
        optimizer = model.make_optimizer()
        cells = torch.randint(0, 65536, (113, 3), device='cuda')
        model.position_embedding(cells).square().mean().backward()
        model.update_learning_rates(optimizer, 1000)
        optimizer.step()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)/'position.pt'
            save_checkpoint(checkpoint, model, optimizer, 1000)
            restored, opt, step, _ = load_checkpoint(checkpoint, 'cuda', resume=True)
            self.assertEqual(step, 1000)
            for current, current_opt in ((model, optimizer), (restored, opt)):
                current_opt.zero_grad(set_to_none=True)
                current.update_learning_rates(current_opt, 1001)
                current.position_embedding(cells).square().mean().backward()
                current_opt.step()
            torch.testing.assert_close(model.octree_mlp.network.params,
                                       restored.octree_mlp.network.params, atol=1e-7, rtol=1e-5)
            restored.update_learning_rates(opt, 5000)
            self.assertAlmostEqual(next(g['lr'] for g in opt.param_groups if g['name']=='octree_mlp'), 0.008)
            model.freeze_coordinates()
            model.eval()
            expected = quantized_hash_reference(model)
            path = Path(temporary)/'position.zip'
            encode_model(model, path, 'cpu', chunk_splats=3)
            decoded, _ = decode_model(path, 'cuda')
            camera = Camera('probe', 16, 16, 12., 12., torch.eye(4, device='cuda'))
            torch.testing.assert_close(render(expected, camera, 35000)['image'],
                                       render(decoded, camera, 35000)['image'], atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
