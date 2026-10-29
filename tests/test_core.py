import math
import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
from arguments import Config
from scene.gaussian_model import SmolGS, gaussian_bits
from utils.graphics_utils import quantize_cells, positional_bits
from utils.octree import encode_octree, decode_octree, huffman_encode, huffman_decode
from utils.arithmetic import encode_gaussian, decode_gaussian
from smolgs.codec import encode_model, decode_model, quantized_hash_reference
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.hashgrid import MixedHashEncoder
from scene.dataset_readers import Camera
from gaussian_renderer import render
from utils.loss_utils import loss_terms


def tiny_config(**changes):
    values = dict(mlp_width=16, octree_mlp_width=8, resolutions_3d=(4, 6), resolutions_2d=(4,), hash_features=2,
                  hash_log2_3d=5,  hash_log2_2d=5,
                  iterations=25, warmup_until=2, densify_until=8,
                  compact_until=12, feature_until=18, lr_max_steps=18,
                  adc_interval=2, hash_backend='torch')
    values.update(changes)
    return Config(**values)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(4)
        np.random.seed(4)

    def make_model(self):
        model = SmolGS(tiny_config())
        points = torch.randn(12, 3)*0.15
        points[:, 2] += 2
        model.initialize(points)
        model.set_scene_extent(torch.tensor([[-100./1.1, 0., 0.], [100./1.1, 0., 0.]]))
        return model


    def test_octree_health_detects_constant_embedding_and_blocked_gradient(self):
        model = self.make_model()
        with torch.no_grad():
            model.octree_mlp[2].weight.zero_()
            model.octree_mlp[2].bias.fill_(-1.)
        before_rng = torch.get_rng_state().clone()
        health = model.octree_health(sample_size=None)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(health['octree_probe_all_points'])
        self.assertEqual(health['octree_second_active_fraction'], 0.)
        self.assertEqual(health['octree_embedding_range'], 0.)
        cells, _ = quantize_cells(model.xyz, model.bounds_min, model.bounds_max, 16)
        model.octree_mlp(positional_bits(cells, 16)).sum().backward()
        self.assertEqual(float(model.octree_mlp[0].weight.grad.norm()), 0.)
        self.assertEqual(float(model.octree_mlp[2].weight.grad.norm()), 0.)
        self.assertGreater(float(model.octree_mlp[4].bias.grad.norm()), 0.)

    def test_compaction_prunes_only_low_average_opacity_and_persistent_invisibility(self):
        for step in (9, 12):
            with self.subTest(step=step):
                model = self.make_model()
                model.config.invisible_patience = 5
                with torch.no_grad():
                    model.log_scales[0].fill_(math.log(1000.))
                    model.vis_count.fill_(2.)
                    model.opacity_accum.fill_(1.)
                    model.last_seen.fill_(step)
                    model.opacity_accum[1] = 0.008  # mean=0.004, below 0.005
                    model.vis_count[2] = 0.; model.last_seen[2] = step-5
                    model.opacity_accum[3] = 0.010  # exactly at threshold: retain
                    model.grad_accum.fill_(100.)  # compaction must never add children
                expected = model.xyz[[0, *range(3, 12)]].detach().clone()
                self.assertEqual(model.density_control(None, torch.zeros(3), step), (0, 2))
                torch.testing.assert_close(model.xyz, expected, rtol=0, atol=0)
                self.assertEqual(model.last_adc_report['pruned_low_opacity'], 1)
                self.assertEqual(model.last_adc_report['pruned_invisible'], 1)

    def test_feature_std_is_only_an_initialization_change(self):
        original = self.make_model()
        wider = copy.deepcopy(original)
        wider.config.feature_init_std = 1.0
        points = original.xyz.detach().clone()
        torch.manual_seed(91); original.initialize(points)
        torch.manual_seed(91); wider.initialize(points)
        torch.testing.assert_close(wider.features, original.features*100)
        for key in ('xyz', 'log_scales', 'bounds_min', 'bounds_max'):
            torch.testing.assert_close(getattr(wider,key),getattr(original,key),rtol=0,atol=0)

    def test_octree_bounds_duplicates_and_huffman(self):
        points = torch.tensor([[0., 0., 0.], [1., 1., 1.], [0.5, 0., 1.]])
        cells, centers = quantize_cells(points, torch.zeros(3), torch.ones(3), 16)
        self.assertTrue((cells[1] == 65535).all())
        blob, order = encode_octree(cells.numpy(), 16)
        restored, depth = decode_octree(blob)
        np.testing.assert_array_equal(restored, cells.numpy()[order])
        self.assertEqual(depth, 16)
        self.assertEqual(positional_bits(torch.tensor([[1, 0, 0]]), 1).tolist(), [[1., 0., 0.]])
        with self.assertRaises(ValueError):
            encode_octree(np.array([[0, 0, 0], [0, 0, 0]]), 16)
        for raw in [b'', b'\0'*1000, bytes(range(256))*5]:
            self.assertEqual(huffman_decode(huffman_encode(raw)), raw)

    def test_conditional_arithmetic_extremes_and_constant(self):
        rng = np.random.RandomState(11)
        for size in [1, 37, 4096]:
            means = rng.normal(size=size)
            stds = np.exp(rng.normal(size=size))
            steps = np.exp(rng.normal(size=size))*0.3
            values = means+rng.normal(size=size)*stds
            blob = encode_gaussian(values, means, stds, steps)
            actual = decode_gaussian(blob, means, stds, steps)
            np.testing.assert_array_equal(np.rint(actual/steps), np.rint(values/steps))
        values = np.array([-250., 0., 250., 0.])
        means = np.zeros(4)
        stds = np.ones(4)*1e-6
        steps = np.ones(4)
        np.testing.assert_array_equal(decode_gaussian(encode_gaussian(values, means, stds, steps),
                                                     means, stds, steps), values)

    def test_screen_adc_uses_xy_visible_and_masked_points(self):
        model = self.make_model()
        model.config.densify_threshold = 6.0
        mask = torch.zeros(len(model.xyz), dtype=torch.bool)
        mask[[1, 4, 8]] = True
        attributes = {'mask': mask, 'opacity': torch.tensor([[0.2], [0.9], [0.6]])}
        visible = torch.tensor([True, False, True])
        projected = torch.zeros(3, 3, requires_grad=True)
        projected.grad = torch.tensor([[3., 4., 1000.], [9., 12., 1000.], [5., 12., 1000.]])
        # A large world gradient must not affect the screen-space decision.
        model.xyz.grad = torch.full_like(model.xyz, 1000.)
        model.collect_statistics(attributes, visible, 6, projected)
        expected = torch.zeros(len(model.xyz)); expected[[1, 8]] = torch.tensor([5., 13.])
        torch.testing.assert_close(model.grad_accum, expected, rtol=0, atol=0)
        self.assertEqual(model.vis_count.tolist(), [0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
        torch.testing.assert_close(model.opacity_accum[[1, 8]], torch.tensor([0.2, 0.6]))
        added, _ = model.density_control(model.make_optimizer(), torch.zeros(3), 6)
        self.assertEqual(added, 1)
        self.assertEqual(len(model.xyz), 13)

    def test_screen_adc_rejects_missing_projected_gradient(self):
        model = self.make_model()
        attrs = {'mask': torch.ones(len(model.xyz), dtype=torch.bool),
                 'opacity': torch.ones(len(model.xyz), 1)}
        visible = torch.ones(len(model.xyz), dtype=torch.bool)
        model.xyz.grad = torch.ones_like(model.xyz)
        with self.assertRaisesRegex(RuntimeError, 'viewspace_points.grad'):
            model.collect_statistics(attrs, visible, 6)
        self.assertEqual(float(model.vis_count.sum()), 0.)

    def test_clone_split_uses_rendered_axes_and_migrates_adam(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                model = self.make_model().to(device)
                optimizer = model.make_optimizer()
                (model.xyz.square().sum()+model.features.square().sum()+model.log_scales.square().sum()).backward()
                optimizer.step()
                with torch.no_grad():
                    model.log_scales.fill_(math.log(0.01))
                    model.log_scales[2].fill_(math.log(2.))
                    model.log_scales[2] += model.xyz.new_tensor([0.5, 0.75, 0.25]).log()
                    model.rotations.copy_(model.xyz.new_tensor([math.sqrt(0.5), 0., 0., math.sqrt(0.5)]).expand_as(model.rotations))
                    model.grad_accum[[1, 2]] = 1.
                    model.vis_count.fill_(1.)
                    model.opacity_accum.fill_(0.5)
                originals = {name:getattr(model,name).detach().clone() for name in ('xyz','features','log_scales')}
                moments = {name:{key:optimizer.state[getattr(model,name)][key].clone()
                                for key in ('exp_avg','exp_avg_sq')} for name in originals}
                normal = model.xyz.new_tensor([[1., 2., 3.], [-2., 1., 0.5]])
                axes = originals['log_scales'][[2,2]].exp()
                local = normal*axes
                expected_offset = torch.stack((-local[:,1],local[:,0],local[:,2]),-1)
                with mock.patch('torch.randn_like', return_value=normal) as sample:
                    self.assertEqual(model.density_control(optimizer, torch.zeros(3,device=device), 6), (3, 1))
                    self.assertEqual(sample.call_count, 1)
                self.assertEqual(len(model.xyz), 14)
                survivors = torch.tensor([0, 1, *range(3, 12)], device=device)
                # Small splats are exact copies; large parents are removed.
                torch.testing.assert_close(model.xyz[11], originals['xyz'][1], rtol=0, atol=0)
                torch.testing.assert_close(model.xyz[12:], originals['xyz'][[2,2]]+expected_offset)
                for name, original in originals.items():
                    parameter = getattr(model,name)
                    torch.testing.assert_close(parameter[:11], original[survivors], rtol=0, atol=0)
                    if name != 'xyz':
                        expected = original[[1,2,2]].clone()
                        if name == 'log_scales':expected[1:] -= math.log(1.6)
                        torch.testing.assert_close(parameter[11:], expected, rtol=0, atol=0)
                    self.assertIs(next(g for g in optimizer.param_groups if g['name']==name)['params'][0], parameter)
                    for key in ('exp_avg','exp_avg_sq'):
                        torch.testing.assert_close(optimizer.state[parameter][key][:11],moments[name][key][survivors],rtol=0,atol=0)
                        self.assertEqual(float(optimizer.state[parameter][key][11:].abs().sum()),0.)
                self.assertEqual(model.last_seen[11:].tolist(), [6.,6.,6.])
                self.assertEqual(model.last_adc_report['cloned'], 1)
                self.assertEqual(model.last_adc_report['split_parents'], 1)

    def test_adc_has_no_size_pruning_even_for_large_visible_opaque_splats(self):
        model = self.make_model()
        model.config.warmup_until = 500
        model.config.densify_until = 15000
        model.config.compact_until = 20000
        with torch.no_grad():
            model.log_scales[0].fill_(math.log(1000.))
            model.xyz[0, 0] = 50000.
            model.vis_count.fill_(1.)
            model.opacity_accum.fill_(0.5)
            model.last_seen.fill_(4000.)
            model.scene_extent.zero_()  # no size criterion needs a camera extent
        original = model.xyz.detach().clone()
        self.assertEqual(model.density_control(None,torch.zeros(3),4000),(0,0))
        torch.testing.assert_close(model.xyz,original,rtol=0,atol=0)

    def test_clone_split_respects_net_point_budget_on_cpu_and_cuda(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                model = self.make_model().to(device)
                for limit, expected in [(12,(0,0)),(13,(1,0))]:
                    with torch.no_grad():
                        model.grad_accum[0] = 1.
                        model.vis_count.fill_(1.)
                        model.opacity_accum.fill_(0.5)
                    self.assertEqual(model.density_control(None,torch.zeros(3,device=device),6,max_splats=limit),expected)
                self.assertEqual(len(model.xyz),13)
                torch.testing.assert_close(model.log_scales[-1],model.log_scales[0],rtol=0,atol=0)
                with torch.no_grad():
                    model.log_scales[1].fill_(math.log(100.))
                    model.grad_accum[1] = 1.
                    model.vis_count.fill_(1.)
                    model.opacity_accum.fill_(0.5)
                self.assertEqual(model.density_control(None,torch.zeros(3,device=device),6,max_splats=14),(2,1))
                self.assertEqual(len(model.xyz),14)
                torch.testing.assert_close(model.log_scales[-2:],model.xyz.new_full((2,3),math.log(100./1.6)))

    def test_clone_split_classifies_rendered_size_at_boundary(self):
        model = self.make_model()
        with torch.no_grad():
            # Large controllers can still represent small physical splats.
            model.log_scales.fill_(math.log(100.))
            model.vis_count.fill_(1.)
            model.opacity_accum.fill_(0.5)
            model.grad_accum[[0,1]] = 1.
        axes = model.xyz.new_tensor([[1.,0.2,0.3],[1.01,0.2,0.3]])
        rotation = model.xyz.new_tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]])
        with mock.patch.object(model,'attributes',return_value={'scales':axes,'rotation':rotation}):
            self.assertEqual(model.density_control(None,torch.zeros(3),6),(3,1))
        self.assertEqual(model.last_adc_report['cloned'],1)
        self.assertEqual(model.last_adc_report['split_parents'],1)

    def test_coordinate_collision_guard_is_transactional(self):
        model = self.make_model()
        optimizer = model.make_optimizer()
        model.xyz.square().sum().backward()
        optimizer.step()
        with torch.no_grad():
            model.xyz.zero_()
            model.xyz[-1].fill_(50000.)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        old_parameter = model.xyz
        old_moment = optimizer.state[model.xyz]['exp_avg'].clone()
        with self.assertRaisesRegex(RuntimeError, 'above the collision limit'):
            model.freeze_coordinates(optimizer)
        self.assertIs(model.xyz, old_parameter)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[model.xyz]['exp_avg'], old_moment, rtol=0, atol=0)

    def test_adc_stage_boundaries_and_strict_gradient_threshold(self):
        for step in (0,500,501,15000,15001,20000,20001):
            with self.subTest(step=step):
                model=self.make_model()
                model.config.warmup_until=500
                model.config.densify_until=15000
                model.config.compact_until=20000
                with torch.no_grad():
                    model.vis_count.fill_(2.)
                    model.opacity_accum.fill_(1.)
                    model.last_seen.fill_(step)
                    model.grad_accum[0]=model.config.densify_threshold*2  # exactly equal, no birth
                    model.grad_accum[1]=model.config.densify_threshold*4
                expected=(1,0) if 500<step<=15000 else (0,0)
                self.assertEqual(model.density_control(None,torch.zeros(3),step),expected)
                if not 500<step<=20000:
                    self.assertEqual(float(model.vis_count.sum()),24.)
                    attrs={'mask':torch.ones(12,dtype=torch.bool),'opacity':torch.ones(12,1)}
                    model.collect_statistics(attrs,torch.ones(12,dtype=torch.bool),step)
                    self.assertEqual(float(model.vis_count.sum()),24.)

    def test_invalid_adc_state_and_all_pruned_do_not_change_topology(self):
        for failure in ('nonfinite','all_pruned'):
            with self.subTest(failure=failure):
                model=self.make_model()
                with torch.no_grad():
                    model.vis_count.fill_(1.)
                    model.opacity_accum.fill_(0.5)
                    if failure=='nonfinite':model.features[3,0]=float('nan')
                    if failure=='all_pruned':model.opacity_accum.zero_()
                parameter=model.xyz
                old=model.xyz.detach().clone()
                state=model.vis_count.clone()
                exception=FloatingPointError if failure=='nonfinite' else RuntimeError
                with self.assertRaises(exception):model.density_control(None,torch.zeros(3),6)
                self.assertIs(model.xyz,parameter)
                torch.testing.assert_close(model.xyz,old,rtol=0,atol=0)
                torch.testing.assert_close(model.vis_count,state,rtol=0,atol=0)


    def test_checkpoint_rejects_missing_or_incompatible_adc_policy(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'incompatible.pt'
            save_checkpoint(path, model, model.make_optimizer(), 4)
            payload = torch.load(path, weights_only=False)
            del payload['adc_policy']
            torch.save(payload, path)
            for resume in (False, True):
                with self.assertRaisesRegex(ValueError, 'Unsupported ADC'):
                    load_checkpoint(path, resume=resume)

    def test_render_backward_supplies_screen_adc_gradient(self):
        backends = [('torch', 'cpu')]
        if torch.cuda.is_available():
            backends.append(('cuda', 'cuda'))
        for backend, device in backends:
            with self.subTest(backend=backend):
                model = self.make_model().to(device)
                camera = Camera('tiny', 16, 16, 14., 14., torch.eye(4)).to(device)
                package = render(model, camera, 6, backend)
                loss, _ = loss_terms(model, package, torch.full((3, 16, 16), 0.25, device=device), 6)
                loss.backward()
                projected = package['viewspace_points']
                self.assertIsNotNone(projected.grad)
                self.assertTrue(torch.isfinite(projected.grad).all())
                self.assertGreater(float(projected.grad[:, :2].abs().sum()), 0.)
                self.assertIsNotNone(model.xyz.grad)
                expected = projected.grad[package['visible'], :2].norm(dim=-1)
                # The accumulator is independent of the world-coordinate grad.
                model.xyz.grad = None
                model.collect_statistics(package['attributes'], package['visible'], 6, projected)
                torch.testing.assert_close(model.grad_accum[package['visible']], expected)

    def test_rate_tail_stability(self):
        x = torch.tensor([-100., 0., 100.], requires_grad=True)
        rates = gaussian_bits(x, torch.zeros(3), torch.ones(3), torch.ones(3))
        self.assertTrue(torch.isfinite(rates).all())
        rates.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_adam_survivors_clones_and_coordinate_freeze(self):
        model = self.make_model()
        optimizer = model.make_optimizer()
        (model.xyz.square().sum()+model.features.square().sum()+model.log_scales.square().sum()).backward()
        optimizer.step()
        old = optimizer.state[model.features]['exp_avg'].clone()
        indices = torch.tensor([2, 0, 2])
        model.replace_topology(indices, optimizer, new_start=2)
        np.testing.assert_array_equal(optimizer.state[model.features]['exp_avg'][:2].numpy(), old[[2, 0]].numpy())
        self.assertTrue((optimizer.state[model.features]['exp_avg'][2] == 0).all())
        removed = model.freeze_coordinates(optimizer)
        self.assertEqual(removed, 1)
        self.assertFalse(model.xyz.requires_grad)
        model.update_learning_rates(optimizer, 20)
        self.assertEqual(next(g['lr'] for g in optimizer.param_groups if g['name'] == 'xyz'), 0)

    def test_five_stage_backward_and_roundtrip_render(self):
        model = self.make_model()
        optimizer = model.make_optimizer()
        camera = Camera('tiny', 16, 16, 14., 14., torch.eye(4))
        target = torch.full((3, 16, 16), 0.25)
        stages = set()
        for step in range(1, model.config.iterations+1):
            stages.add(model.config.stage(step))
            if step == model.config.feature_until+1:
                model.freeze_coordinates(optimizer)
            optimizer.zero_grad(set_to_none=True)
            package = render(model, camera, step, 'torch')
            loss, _ = loss_terms(model, package, target, step)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            model.collect_statistics(package['attributes'],package['visible'],step,package['viewspace_points'])
            optimizer.step()
            if step % model.config.adc_interval == 0:
                model.density_control(optimizer,camera.center,step,max_splats=64)
        self.assertEqual(len(stages), 5)
        model.eval()
        before = render(quantized_hash_reference(model), camera, 25, 'torch')['image']
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.smolgs.zip'
            encode_model(model, path, 'cpu', chunk_splats=3)
            restored, report = decode_model(path)
            self.assertTrue(bool(restored.decoded))
            after = render(restored, camera, 25, 'torch')['image']
            torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-6)

    def test_archive_roundtrip_with_oversized_single_splat_alphabet(self):
        # Regress the full-size Truck export: one splat can exceed HAC's
        # int16 alphabet across channels, even after splitting to one row.
        for codec in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as temporary:
                model = self.make_model().to(codec if codec == 'cuda' else 'cpu')
                with torch.no_grad():
                    for parameter in model.entropy_mlp.parameters():
                        parameter.zero_()  # Fixed feature step=1; geometry is stored losslessly.
                    model.features[0, :2] = model.features.new_tensor([-2_000_000., 2_000_000.])
                    model.log_scales[0] = model.log_scales.new_tensor([0.001, 4000., 1.]).log()
                    model.freeze_coordinates()
                    model.eval()
                    _, _, qf = model.prior(model.xyz)
                    expected_f = torch.round(model.features/qf)*qf
                    expected_s = model.log_scales.exp()
                    path = Path(temporary)/'model.smolgs.zip'
                    report = encode_model(model, path, codec)
                    self.assertGreater(report['segmented_attributes'], 0)
                    restored, _ = decode_model(path, str(model.xyz.device))
                    torch.testing.assert_close(restored.features, expected_f, rtol=0, atol=0)
                    torch.testing.assert_close(restored.log_scales.exp(), expected_s, rtol=1e-6, atol=1e-7)

    def test_archive_quantization_context_independent_of_stream_chunks(self):
        # Simulate a one-ULP GEMM difference between scene and chunk batches.
        # At f/q=2.5 this changes the integer symbol, not just its float value.
        for codec in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as temporary:
                model = self.make_model().to(codec if codec == 'cuda' else 'cpu')
                with torch.no_grad():
                    model.features.fill_(2.5)
                    model.freeze_coordinates()
                model.eval()
                count = len(model.xyz)

                def shape_sensitive_prior(candidate, xyz):
                    n, d = len(xyz), candidate.config.feature_dim
                    q = xyz.new_tensor(1.)
                    if n != count:
                        q = torch.nextafter(q, xyz.new_tensor(0.))
                    return (xyz.new_zeros(n, d), xyz.new_ones(n, d), q.expand(n, 1))

                with mock.patch.object(SmolGS, 'prior', shape_sensitive_prior):
                    for chunk_size in (3, count):
                        path = Path(temporary)/f'model_{chunk_size}.zip'
                        encode_model(model, path, codec, chunk_splats=chunk_size)
                        restored, _ = decode_model(path, str(model.xyz.device))
                        torch.testing.assert_close(restored.features, torch.full_like(model.features, 2.),
                                                   rtol=0, atol=0)

    def test_checkpoint_resume_replays_optimizer_and_rng(self):
        model = self.make_model()
        optimizer = model.make_optimizer()
        model.features.sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.pt'
            save_checkpoint(path, model, optimizer, 4)
            expected = (random.random(), np.random.rand(), torch.rand(3))
            restored, opt, step, _ = load_checkpoint(path, resume=True)
            actual = (random.random(), np.random.rand(), torch.rand(3))
            self.assertEqual(expected[:2], actual[:2])
            torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
            self.assertEqual(step, 4)
            for candidate, optim in [(model, optimizer), (restored, opt)]:
                optim.zero_grad(set_to_none=True)
                candidate.features.square().sum().backward()
                optim.step()
            torch.testing.assert_close(model.features, restored.features, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_native_binary_hash_matches_torch_reference(self):
        reference = MixedHashEncoder(tiny_config()).cuda()
        accelerated = MixedHashEncoder(tiny_config(hash_backend='cuda')).cuda()
        accelerated.load_state_dict(reference.state_dict())
        points = torch.cat([torch.rand(19, 3), torch.zeros(1, 3), torch.ones(1, 3)]).cuda()
        actual, expected = accelerated(points), reference(points)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-4)
        actual.sum().backward(); expected.sum().backward()
        for name, parameter in accelerated.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad,
                                       atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
