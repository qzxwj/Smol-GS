"""Static geometry without VQ: gradients, topology, resume and lossless export."""
import copy
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from arguments import Config, config_from_args
from gaussian_renderer import render
from scene.dataset_readers import Camera
from scene.gaussian_model import SmolGS, gaussian_bits
from smolgs.codec import encode_model, decode_model, quantized_hash_reference
from test_core import tiny_config
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.graphics_utils import morton_codes, quantize_cells, quaternion_matrix
from utils.loss_utils import loss_terms


class ExplicitGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(63)
        random.seed(63)
        np.random.seed(63)

    def devices(self):
        return ['cpu']+(['cuda'] if torch.cuda.is_available() else [])

    def model(self, device='cpu', **changes):
        config = tiny_config(geometry_mode='explicit', **changes)
        model = SmolGS(config, device=device)
        points = torch.randn(12, 3, device=device)*0.12
        points[:, 2] += 2
        model.initialize(points)
        model.set_scene_extent(torch.tensor([[-100./1.1, 0., 0.], [100./1.1, 0., 0.]], device=device))
        return model

    def anisotropic(self, model):
        with torch.no_grad():
            model.log_scales.copy_(model.xyz.new_tensor([.08, .15, .04]).log().expand_as(model.log_scales))
            model.rotations.copy_(model.xyz.new_tensor([.8, .2, -.3, .4]).expand_as(model.rotations))

    def test_config_cli_and_effective_optimizers(self):
        from train import parser
        args = parser().parse_args(['-s', '/tmp/data', '-m', '/tmp/output',
                                    '--geometry-mode', 'explicit', '--explicit-rotation-lr', '.003'])
        config = config_from_args(args)
        self.assertEqual(config.geometry_mode, 'explicit')
        model = SmolGS(config)
        rates = config.optimizer_rates()
        self.assertNotIn('scale_mlp', rates)
        self.assertNotIn('rotation_mlp', rates)
        optimizer = model.make_optimizer()
        for step in (0, 20001, 35000):
            model.update_learning_rates(optimizer, step)
            groups = {g['name']: g for g in optimizer.param_groups}
            self.assertAlmostEqual(groups['log_scales']['lr'], .005)
            self.assertAlmostEqual(groups['rotations']['lr'], .003)
        parameters = [p for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(parameters), len(set(parameters)))
        self.assertEqual(set(parameters), set(model.parameters()))
        for changes in ({'geometry_mode': 'bad'}, {'geometry_mode': 'neural'}, {'explicit_rotation_lr': 0}, {'explicit_scaling_lr': float('nan')}):
            with self.assertRaises(ValueError):
                tiny_config(**changes)

    def test_default_geometry_and_initialization_have_no_geometry_networks(self):
        for appearance, octree, device in [('torch', 'torch', 'cpu')]+(
                [('tcnn', 'tcnn', 'cuda')] if torch.cuda.is_available() else []):
            with self.subTest(appearance=appearance):
                options = dict(appearance_backend=appearance, octree_backend=octree,
                               octree_mlp_width=64 if octree == 'tcnn' else 8)
                torch.manual_seed(19)
                default = SmolGS(tiny_config(**options), device=device)
                torch.manual_seed(19)
                repeated = SmolGS(tiny_config(**options), device=device)
                self.assertEqual(default.config.geometry_mode, 'explicit')
                self.assertFalse(hasattr(default, 'rotation_mlp'))
                self.assertFalse(hasattr(default, 'scale_mlp'))
                self.assertEqual(default.entropy_mlp[-1].out_features, 2*default.config.feature_dim+1)
                for name, value in default.state_dict().items():
                    torch.testing.assert_close(value, repeated.state_dict()[name], rtol=0, atol=0)
                points = torch.tensor([[0., 0., 2.], [.1, 0., 2.], [0., .2, 2.], [.3, .2, 2.]], device=device)
                default.initialize(points)
                torch.testing.assert_close(default.rotations, points.new_tensor([1., 0., 0., 0.]).expand(4, 4))
                torch.testing.assert_close(default.attributes(points.new_zeros(3), 1)['scales'], default.log_scales.exp())

    def test_geometry_is_view_feature_stage_and_noise_independent(self):
        for device in self.devices():
            with self.subTest(device=device):
                model = self.model(device)
                self.anisotropic(model)
                scale = model.log_scales.exp().detach()
                rotation = F.normalize(model.rotations.detach(), dim=-1)
                covariance = quaternion_matrix(rotation) @ torch.diag_embed(scale.square()) @ quaternion_matrix(rotation).transpose(1, 2)
                for training in (True, False):
                    model.train(training)
                    for step in (1, 8, 12, 13, 18, 19, 25):
                        for center in (model.xyz.new_zeros(3), model.xyz.new_tensor([2., -1., .5])):
                            attrs = model.attributes(center, step)
                            torch.testing.assert_close(attrs['scales'], scale, rtol=0, atol=0)
                            torch.testing.assert_close(attrs['rotation'], rotation, rtol=0, atol=0)
                            r = quaternion_matrix(attrs['rotation'])
                            torch.testing.assert_close(r @ torch.diag_embed(attrs['scales'].square()) @ r.transpose(1, 2), covariance)
                with torch.no_grad():
                    model.features.normal_(mean=100., std=10.)
                torch.testing.assert_close(model.attributes(center, 25)['scales'], scale, rtol=0, atol=0)

    def test_masks_empty_batches_and_degenerate_quaternion(self):
        for device in self.devices():
            model = self.model(device)
            self.anisotropic(model)
            for count in (0, 1, 3):
                mask = torch.arange(len(model.xyz), device=device) < count
                attrs = model.attributes(model.xyz.new_zeros(3), 25, mask)
                self.assertEqual(attrs['scales'].shape, (count, 3))
                self.assertEqual(attrs['rotation'].shape, (count, 4))
                self.assertTrue(torch.isfinite(attrs['rate']))
                torch.testing.assert_close(attrs['scales'], model.log_scales[mask].exp())
            with torch.no_grad():
                model.rotations[0].zero_()
                model.rotations[1].fill_(1e-20)
            attrs = model.attributes(model.xyz.new_zeros(3), 25)
            torch.testing.assert_close(attrs['rotation'][:2], model.xyz.new_tensor([1., 0., 0., 0.]).expand(2, 4))
            attrs['rotation'].sum().backward()
            self.assertTrue(torch.isfinite(model.rotations.grad).all())

    def test_geometry_gradient_and_feature_only_rate(self):
        for device in self.devices():
            model = self.model(device)
            self.anisotropic(model)
            center = model.xyz.new_zeros(3)
            torch.manual_seed(5)
            attrs = model.attributes(center, 25)
            torch.manual_seed(5)
            mf, sf, qf = model.prior(model.xyz)
            noisy_features = model.features+(torch.rand_like(model.features)-.5)*qf
            expected_rate = gaussian_bits(noisy_features, mf, sf, qf, model.config.rate_bin_half_width).sum(-1).mean()
            torch.testing.assert_close(attrs['rate'], expected_rate, rtol=0, atol=0)
            scale_grad, rot_grad = torch.autograd.grad(attrs['rate'], (model.log_scales, model.rotations),
                                                     allow_unused=True, retain_graph=True)
            self.assertIsNone(scale_grad)
            self.assertIsNone(rot_grad)
            objective = attrs['scales'].sum()+attrs['rotation'][:, 2].sum()
            grads = torch.autograd.grad(objective, (model.log_scales, model.rotations, model.features), allow_unused=True)
            torch.testing.assert_close(grads[0], model.log_scales.exp())
            self.assertGreater(float(grads[1].abs().sum()), 0.)
            self.assertIsNone(grads[2])
            self.assertTrue(torch.isfinite(grads[1]).all())

    def test_render_backward_updates_geometry_on_cpu_and_cuda(self):
        for device in self.devices():
            with self.subTest(device=device):
                model = self.model(device)
                self.anisotropic(model)
                optimizer = model.make_optimizer()
                camera = Camera('probe', 20, 16, 16., 16., torch.eye(4, device=device))
                old = {name: getattr(model, name).detach().clone() for name in ('log_scales', 'rotations')}
                package = render(model, camera, 6, 'cuda' if device == 'cuda' else 'torch')
                target = torch.linspace(0., .7, 20, device=device)[None, None, :].expand(3, 16, 20)
                loss, _ = loss_terms(model, package, target, 6)
                loss.backward()
                for name in old:
                    gradient = getattr(model, name).grad
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(float(gradient.abs().sum()), 0.)
                self.assertGreater(float(package['viewspace_points'].grad[:, :2].abs().sum()), 0.)
                optimizer.step()
                for name in old:
                    self.assertFalse(torch.equal(old[name], getattr(model, name)))

    def test_clone_split_prune_and_all_adam_moments_remain_aligned(self):
        for device in self.devices():
            with self.subTest(device=device):
                model = self.model(device)
                groups = model.make_optimizer().param_groups
                for group in groups: group['amsgrad'] = True
                optimizer = torch.optim.Adam(groups, amsgrad=True)
                sum(getattr(model, name).square().sum() for name in model.point_parameter_names).backward()
                optimizer.step()
                with torch.no_grad():
                    model.log_scales.fill_(math.log(.01))
                    model.log_scales[2].copy_(model.xyz.new_tensor([2., 3., 4.]).log())
                    model.rotations.copy_(torch.arange(48, device=device).reshape(12, 4)+1.)
                    model.rotations[2].copy_(model.xyz.new_tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)]))
                    model.vis_count.fill_(1.)
                    model.opacity_accum.fill_(.5)
                    model.last_seen.fill_(6.)
                    model.opacity_accum[3] = 0.
                    model.grad_accum[[1, 2]] = 1.
                originals = {name: getattr(model, name).detach().clone() for name in model.point_parameter_names}
                old_parameters = {name: getattr(model, name) for name in originals}
                states = {name: copy.deepcopy(optimizer.state[p]) for name, p in old_parameters.items()}
                normal = model.xyz.new_tensor([[1., 2., 3.], [-2., 1., .5]])
                local = normal*model.xyz.new_tensor([2., 3., 4.])
                offset = torch.stack([-local[:, 1], local[:, 0], local[:, 2]], -1)
                with mock.patch('torch.randn_like', return_value=normal):
                    self.assertEqual(model.density_control(optimizer, model.xyz.new_zeros(3), 6), (3, 2))
                survivors = torch.tensor([0, 1, *range(4, 12)], device=device)
                indices = torch.tensor([0, 1, *range(4, 12), 1, 2, 2], device=device)
                for name, original in originals.items():
                    parameter = getattr(model, name)
                    expected = original[indices].clone()
                    if name == 'xyz': expected[-2:] += offset
                    if name == 'log_scales': expected[-2:] -= math.log(1.6)
                    torch.testing.assert_close(parameter, expected)
                    self.assertNotIn(old_parameters[name], optimizer.state)
                    self.assertIs(next(g for g in optimizer.param_groups if g['name'] == name)['params'][0], parameter)
                    state = optimizer.state[parameter]
                    torch.testing.assert_close(state['step'], states[name]['step'])
                    for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                        torch.testing.assert_close(state[key][:10], states[name][key][survivors], rtol=0, atol=0)
                        self.assertEqual(float(state[key][10:].abs().sum()), 0.)
                optimizer.zero_grad(set_to_none=True)
                model.rotations[:, 1].sum().backward()
                optimizer.step()
                self.assertGreater(float(optimizer.state[model.rotations]['exp_avg'][10:].abs().sum()), 0.)

    def test_coordinate_freeze_reorders_and_deduplicates_geometry_and_moments(self):
        for device in self.devices():
            model = self.model(device, octree_depth=3, coordinate_depth=3)
            with torch.no_grad():
                model.xyz.copy_(model.xyz.new_tensor([[1., 0., 2.], [0., 0., 2.], [2., 1., 3.], [1., 0., 2.]]*3))
                model.rotations.copy_(torch.arange(48, device=device).reshape(12, 4)+1.)
                model.log_scales.copy_(torch.arange(36, device=device).reshape(12, 3)*.01-3.)
            model.config.coordinate_max_collision_fraction = .9
            optimizer = model.make_optimizer()
            sum(getattr(model, name).square().sum() for name in model.point_parameter_names).backward()
            optimizer.step()
            # Restore exact duplicate locations after the optimizer update.
            with torch.no_grad():
                model.xyz.copy_(model.xyz.new_tensor([[1., 0., 2.], [0., 0., 2.], [2., 1., 3.], [1., 0., 2.]]*3))
            cells, centers = quantize_cells(model.xyz, model.xyz.amin(0), model.xyz.amax(0), 3)
            codes = morton_codes(cells, 3)
            order = torch.argsort(codes, stable=True)
            keep = torch.ones(len(order), dtype=torch.bool, device=device)
            keep[1:] = codes[order][1:] != codes[order][:-1]
            indices = order[keep]
            originals = {name: getattr(model, name).detach().clone() for name in model.point_parameter_names}
            moments = {name: optimizer.state[getattr(model, name)]['exp_avg'].clone() for name in originals}
            self.assertEqual(model.freeze_coordinates(optimizer), 9)
            for name in originals:
                expected = centers[indices] if name == 'xyz' else originals[name][indices]
                torch.testing.assert_close(getattr(model, name), expected, rtol=0, atol=0)
                torch.testing.assert_close(optimizer.state[getattr(model, name)]['exp_avg'], moments[name][indices], rtol=0, atol=0)
            self.assertFalse(model.xyz.requires_grad)
            self.assertTrue(model.rotations.requires_grad)
            model.update_learning_rates(optimizer, 25)
            self.assertEqual(next(g['lr'] for g in optimizer.param_groups if g['name'] == 'xyz'), 0.)

    def test_compaction_and_invalid_rotation_are_transactional(self):
        model = self.model()
        with torch.no_grad():
            model.rotations.copy_(torch.arange(48).reshape(12, 4)+1.)
            model.vis_count.fill_(1.)
            model.opacity_accum.fill_(.5)
            model.last_seen.fill_(10.)
            model.opacity_accum[2] = 0.
        original = model.rotations.detach().clone()
        self.assertEqual(model.density_control(None, torch.zeros(3), 10), (0, 1))
        torch.testing.assert_close(model.rotations, original[[0, 1, *range(3, 12)]], rtol=0, atol=0)
        with torch.no_grad(): model.rotations[0, 0] = float('nan')
        old = model.xyz
        with self.assertRaisesRegex(FloatingPointError, 'nonfinite'):
            model.density_control(None, torch.zeros(3), 10)
        self.assertIs(model.xyz, old)

    def test_resume_replays_geometry_noise_optimizer_and_frozen_stage(self):
        for device in self.devices():
            for frozen in (False, True):
                with self.subTest(device=device, frozen=frozen), tempfile.TemporaryDirectory() as temporary:
                    model = self.model(device)
                    self.anisotropic(model)
                    optimizer = model.make_optimizer()
                    camera = Camera('probe', 16, 16, 14., 14., torch.eye(4, device=device))
                    backend = 'cuda' if device == 'cuda' else 'torch'

                    def update(candidate, opt):
                        opt.zero_grad(set_to_none=True)
                        candidate.update_learning_rates(opt, 20)
                        package = render(candidate, camera, 20, backend)
                        loss, _ = loss_terms(candidate, package, torch.full_like(package['image'], .25), 20)
                        loss.backward()
                        opt.step()

                    update(model, optimizer)
                    if frozen: model.freeze_coordinates(optimizer)
                    path = Path(temporary)/'checkpoint.pt'
                    save_checkpoint(path, model, optimizer, 19)
                    saved_state = {name: value.clone() for name, value in model.state_dict().items()}
                    saved_optimizer = {name: copy.deepcopy(optimizer.state[getattr(model, name)])
                                       for name in model.point_parameter_names}
                    update(model, optimizer)
                    expected_rng = (random.random(), np.random.rand(), torch.rand(3))
                    restored, opt, step, _ = load_checkpoint(path, device, resume=True)
                    self.assertEqual(step, 19)
                    self.assertEqual(restored.config.geometry_mode, 'explicit')
                    # Loading itself must be exact on both devices. CUDA's
                    # atomic backward reductions need not replay bit for bit.
                    for name, value in saved_state.items():
                        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
                    for name, state in saved_optimizer.items():
                        for key, value in state.items():
                            torch.testing.assert_close(value, opt.state[getattr(restored, name)][key], rtol=0, atol=0)
                    update(restored, opt)
                    actual_rng = (random.random(), np.random.rand(), torch.rand(3))
                    self.assertEqual(expected_rng[:2], actual_rng[:2])
                    torch.testing.assert_close(expected_rng[2], actual_rng[2], rtol=0, atol=0)
                    tolerance = dict(rtol=2e-6, atol=1e-8) if device == 'cuda' else dict(rtol=0, atol=0)
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(value, restored.state_dict()[name], **tolerance)
                    for name in model.point_parameter_names:
                        for key, value in optimizer.state[getattr(model, name)].items():
                            torch.testing.assert_close(value, opt.state[getattr(restored, name)][key], **tolerance)

    def test_explicit_checkpoint_and_archive_cannot_masquerade_as_neural(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.pt'
            save_checkpoint(path, model, model.make_optimizer(), 4)
            payload = torch.load(path, weights_only=False)
            payload['geometry_mode'] = 'neural'
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'geometry mode/config mismatch'):
                load_checkpoint(path, resume=True)
            model.freeze_coordinates()
            path = Path(temporary)/'model.zip'
            encode_model(model, path)
            with zipfile.ZipFile(path) as archive:
                files = {name: archive.read(name) for name in archive.namelist()}
            meta = json.loads(files['metadata.json'])
            meta['geometry_mode'] = 'neural'
            files['metadata.json'] = json.dumps(meta).encode()
            with zipfile.ZipFile(path, 'w') as archive:
                for name, data in files.items(): archive.writestr(name, data)
            with self.assertRaisesRegex(ValueError, 'geometry mode/config mismatch'):
                decode_model(path)

    def test_training_entrypoint_resume_and_final_export(self):
        from PIL import Image
        from train import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'scene'
            sparse = source/'sparse/0'
            sparse.mkdir(parents=True)
            (source/'images').mkdir()
            (sparse/'cameras.txt').write_text('1 PINHOLE 16 16 14 14 8 8\n')
            image_rows = []
            for index in range(9):
                name = f'{index:03d}.png'
                Image.fromarray(np.full((16, 16, 3), 64+index, dtype=np.uint8)).save(source/'images'/name)
                image_rows.append(f'{index+1} 1 0 0 0 {(index-4)*.02} 0 0 1 {name}\n\n')
            (sparse/'images.txt').write_text(''.join(image_rows))
            points = self.model().xyz.detach().numpy()
            (sparse/'points3D.txt').write_text(''.join(
                f'{i+1} {x} {y} {z} 128 128 128 0\n' for i, (x, y, z) in enumerate(points)))
            config = tiny_config(geometry_mode='explicit', densify_threshold=1e6)
            config_path = root/'config.json'
            config_path.write_text(json.dumps(config.to_dict()))
            common = ['-s', str(source), '--config', str(config_path), '--device', 'cpu',
                      '--renderer', 'torch', '--codec', 'cpu', '--test-iterations',
                      '--log-interval', '25', '--checkpoint-interval', '25']
            resumed_dir, continuous_dir = root/'resumed', root/'continuous'
            with redirect_stdout(io.StringIO()):
                main([*common, '-m', str(resumed_dir), '--stop-after', '12'])
                checkpoint = resumed_dir/'checkpoint_latest.pt'
                with self.assertRaisesRegex(ValueError, 'Cannot change geometry mode'):
                    main([*common, '-m', str(resumed_dir), '--resume', str(checkpoint), '--geometry-mode', 'neural'])
                main([*common, '-m', str(resumed_dir), '--resume', str(checkpoint)])
                main([*common, '-m', str(continuous_dir)])
            resumed, _, step, meta = load_checkpoint(resumed_dir/'checkpoint_latest.pt')
            continuous, _, _, _ = load_checkpoint(continuous_dir/'checkpoint_latest.pt')
            self.assertEqual(step, 25)
            self.assertEqual(meta['geometry_quantization'], 'none')
            for name, value in continuous.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
            summary = json.loads((resumed_dir/'summary.json').read_text())
            self.assertTrue(summary['all_stages_completed'])
            self.assertEqual(summary['compression']['geometry_storage'], 'lossless-fp32')
            decoded, _ = decode_model(resumed_dir/'model.smolgs.zip')
            torch.testing.assert_close(decoded.rotations, resumed.rotations, rtol=0, atol=0)

    def test_five_stages_and_lossless_geometry_roundtrip_on_all_backends(self):
        for device, native in [('cpu', False)]+([('cuda', False), ('cuda', True)] if torch.cuda.is_available() else []):
            with self.subTest(device=device, native=native), tempfile.TemporaryDirectory() as temporary:
                model = self.model(device, appearance_backend='tcnn' if native else 'torch',
                                   octree_backend='tcnn' if native else 'torch',
                                   hash_backend='cuda' if native else 'torch',
                                   octree_mlp_width=64 if native else 8)
                optimizer = model.make_optimizer()
                camera = Camera('probe', 16, 16, 14., 14., torch.eye(4, device=device))
                backend = 'cuda' if device == 'cuda' else 'torch'
                stages = set()
                for step in range(1, 26):
                    stages.add(model.config.stage(step))
                    if step == 13:
                        model.hash_min.copy_(model.bounds_min)
                        model.hash_max.copy_(model.bounds_max)
                    if step == 19: model.freeze_coordinates(optimizer)
                    optimizer.zero_grad(set_to_none=True)
                    model.update_learning_rates(optimizer, step)
                    package = render(model, camera, step, backend)
                    loss, _ = loss_terms(model, package, torch.full_like(package['image'], .25), step)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    model.collect_statistics(package['attributes'], package['visible'], step, package['viewspace_points'])
                    optimizer.step()
                    if step % 2 == 0:
                        model.density_control(optimizer, camera.center, step, max_splats=24)
                self.assertEqual(len(stages), 5)
                model.eval()
                reference = quantized_hash_reference(model)
                # Exercise the encoder's own ordering, not just freeze order.
                reverse = torch.arange(len(model.xyz)-1, -1, -1, device=device)
                model.replace_topology(reverse)
                for codec in ['cpu']+(['cuda'] if device == 'cuda' else []):
                    path = Path(temporary)/f'model_{codec}.zip'
                    report = encode_model(model, path, codec, chunk_splats=3)
                    self.assertEqual(report['geometry_storage'], 'lossless-fp32')
                    with zipfile.ZipFile(path) as archive:
                        self.assertFalse(any(name.startswith('scales/') for name in archive.namelist()))
                        weights = torch.load(io.BytesIO(archive.read('networks.pt')), weights_only=False)
                        self.assertFalse(any(name.startswith(('scale_mlp.', 'rotation_mlp.')) for name in weights))
                    decoded, _ = decode_model(path, device)
                    for name in ('log_scales', 'rotations'):
                        torch.testing.assert_close(getattr(decoded, name), getattr(reference, name), rtol=0, atol=0)
                    before = render(reference, camera, 25, backend)
                    after = render(decoded, camera, 25, backend)
                    for name in ('xyz', 'scales', 'rotation', 'color', 'opacity'):
                        torch.testing.assert_close(before['attributes'][name], after['attributes'][name], rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(before['image'], after['image'], rtol=2e-5, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
