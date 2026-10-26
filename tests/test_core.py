import io
import json
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
from smolgs.codec import encode_model, decode_model
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.hashgrid import HashGrid
from scene.dataset_readers import Camera
from gaussian_renderer import render
from utils.loss_utils import loss_terms
from scripts.audit_compaction import effective_candidate_sizes, readonly_view
from scripts.repair_dead_octree import repair
from scripts.audit_octree_embedding_effect import embedding_mean, constant_embedding, fixed_embedding_table


def tiny_config(**changes):
    values = dict(mlp_width=16, octree_mlp_width=8, resolutions_3d=(4, 6),
                  resolutions_2d=(4,), hash_log2_3d=5, hash_log2_2d=4,
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

    def test_octree_initialization_preserves_other_weights_and_rng(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                torch.manual_seed(22)
                original = SmolGS(tiny_config(), device=device)
                expected_rng = torch.get_rng_state().clone()
                expected_cuda = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
                torch.manual_seed(22)
                repaired = SmolGS(tiny_config(octree_init='kaiming'), device=device)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                if expected_cuda is not None:
                    self.assertTrue(torch.equal(expected_cuda, torch.cuda.get_rng_state()))
                for key, value in original.state_dict().items():
                    if not key.startswith('octree_mlp.'):
                        torch.testing.assert_close(value, repaired.state_dict()[key], rtol=0, atol=0)
                self.assertFalse(torch.equal(original.octree_mlp[0].weight, repaired.octree_mlp[0].weight))
                cells = torch.randint(0, 65536, (128, 3), device=device)
                embedding = repaired.octree_mlp(positional_bits(cells, 16))
                self.assertGreater(float((embedding.amax(0)-embedding.amin(0)).max()), 0.)
                embedding.square().sum().backward()
                self.assertGreater(float(repaired.octree_mlp[0].weight.grad.norm()), 0.)

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

    def test_compaction_skips_only_size_rule_after_configured_end(self):
        model = self.make_model()
        model.config.adc_size_prune_from = 0
        model.config.adc_size_prune_until = model.config.densify_until
        model.config.invisible_patience = 5
        model.scene_extent.fill_(10.)
        with torch.no_grad():
            model.log_scales.fill_(math.log(0.01))
            model.log_scales[0].fill_(math.log(10.))
            model.vis_count.fill_(1.)
            model.opacity_accum.fill_(0.5)
            model.last_seen.fill_(10.)
            model.opacity_accum[1] = 0.001
            model.vis_count[2] = 0.; model.last_seen[2] = 0.
            model.features[3, 0] = float('nan')
        before = copy.deepcopy(model)
        before.density_control(None, torch.zeros(3), 6)
        self.assertEqual(before.last_adc_report['pruned_oversized'], 1)
        oversized_xyz = model.xyz[0].detach().clone()
        added, pruned = model.density_control(None, torch.zeros(3), 10)
        self.assertEqual((added, pruned), (0, 3))
        self.assertEqual(model.last_adc_report['pruned_oversized'], 0)
        self.assertEqual(model.last_adc_report['pruned_nonfinite'], 1)
        torch.testing.assert_close(model.xyz[0], oversized_xyz)
        self.assertTrue(torch.isfinite(model.features).all())

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

    def test_leaky_diagnostic_preserves_negative_gradient_and_checkpoint(self):
        model = self.make_model()
        model.config.octree_activation = 'leaky'
        model.octree_mlp[1] = torch.nn.LeakyReLU(0.01)
        model.octree_mlp[3] = torch.nn.LeakyReLU(0.01)
        with torch.no_grad():
            model.octree_mlp[0].weight.fill_(0.01)
            model.octree_mlp[0].bias.zero_()
            model.octree_mlp[2].weight.fill_(0.01)
            model.octree_mlp[2].bias.fill_(-10.)
            model.octree_mlp[4].weight.fill_(0.01)
        cells, _ = quantize_cells(model.xyz, model.bounds_min, model.bounds_max, 16)
        expected = model.octree_mlp(positional_bits(cells,16))
        expected.sum().backward()
        self.assertGreater(float(model.octree_mlp[0].weight.grad.norm()), 0.)
        health = model.octree_health(sample_size=None)
        self.assertEqual(health['octree_second_active_fraction'], 0.)
        self.assertEqual(health['octree_hidden_nonzero_point_fraction'], 1.)
        self.assertGreater(health['octree_embedding_range'], 0.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            save_checkpoint(path,model,None,4)
            restored, _, _, _ = load_checkpoint(path)
            self.assertEqual(restored.config.octree_activation,'leaky')
            torch.testing.assert_close(restored.octree_mlp(positional_bits(cells,16)),expected)

    def test_centered_affine_keeps_initial_function_and_checkpoint(self):
        torch.manual_seed(22);original = SmolGS(tiny_config())
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(22);centered = SmolGS(tiny_config(octree_input_center=0.5))
        self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))
        for key,value in original.state_dict().items():
            if key != 'octree_mlp.0.bias':
                torch.testing.assert_close(value,centered.state_dict()[key],rtol=0,atol=0)
        cells = torch.randint(0,65536,(100,3))
        torch.testing.assert_close(original.position_embedding(cells),centered.position_embedding(cells),atol=1e-6,rtol=1e-5)
        centered.position_embedding(cells).square().sum().backward()
        self.assertGreater(float(centered.octree_mlp[0].weight.grad.norm()),0.)
        # The centered map can be folded back into an ordinary first affine.
        folded = copy.deepcopy(centered.octree_mlp)
        with torch.no_grad():folded[0].bias.sub_(0.5*folded[0].weight.sum(-1))
        torch.testing.assert_close(folded(positional_bits(cells,16)),centered.position_embedding(cells),atol=1e-6,rtol=1e-5)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';save_checkpoint(path,centered,None,4)
            restored,_,_,_=load_checkpoint(path)
            self.assertEqual(restored.config.octree_input_center,0.5)
            torch.testing.assert_close(restored.position_embedding(cells),centered.position_embedding(cells))
            points = torch.randn(12,3)*0.15;points[:,2]+=2
            centered.initialize(points);centered.freeze_coordinates();centered.eval()
            expected = centered.attributes(torch.zeros(3),25)
            archive = Path(directory)/'centered.zip'
            encode_model(centered,archive,'cpu')
            decoded,_ = decode_model(archive)
            self.assertEqual(decoded.config.octree_input_center,0.5)
            actual = decoded.attributes(torch.zeros(3),25)
            for key in ('xyz','opacity','color','rotation','scales'):
                torch.testing.assert_close(actual[key],expected[key],atol=2e-6,rtol=2e-6)

    def test_mean_embedding_intervention_is_readonly_and_preserves_constant_model(self):
        model=self.make_model()
        with torch.no_grad():
            model.octree_mlp[2].weight.zero_()
            model.octree_mlp[2].bias.fill_(-1.)
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        rng=torch.get_rng_state().clone()
        expected=model.attributes(torch.zeros(3),0)
        mean=embedding_mean(model)
        self.assertNotIn('position_embedding',model.__dict__)
        with constant_embedding(model,mean):
            actual=model.attributes(torch.zeros(3),0)
            for key,value in expected.items():
                torch.testing.assert_close(actual[key],value,rtol=0,atol=0)
        self.assertNotIn('position_embedding',model.__dict__)
        with self.assertRaisesRegex(RuntimeError,'restore'):
            with constant_embedding(model,mean):
                raise RuntimeError('restore')
        self.assertNotIn('position_embedding',model.__dict__)
        for key,value in state.items():
            torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_both_affine_centers_preserve_initial_function_rng_and_archive(self):
        points=torch.tensor([[i/10,(i%3)/10,2.+(i%5)/10] for i in range(12)])
        torch.manual_seed(61);original=SmolGS(tiny_config());original.initialize(points)
        rng=torch.get_rng_state().clone()
        torch.manual_seed(61)
        centered=SmolGS(tiny_config(octree_input_center_mode='sfm',octree_hidden_center_mode='sfm'))
        centered.initialize(points)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(centered.octree_mlp.use_hidden_center)
        self.assertEqual(len(centered.config.octree_hidden_center),8)
        self.assertEqual(len(centered.config.octree_input_mean),48)
        self.assertNotIn('position_input_center',centered.state_dict())
        self.assertNotIn('octree_mlp.hidden_center',centered.state_dict())
        self.assertEqual(set(original.state_dict()),set(centered.state_dict()))
        for key,value in original.state_dict().items():
            if key not in ('octree_mlp.0.bias','octree_mlp.2.bias'):
                torch.testing.assert_close(value,centered.state_dict()[key],rtol=0,atol=0)
        cells=torch.randint(0,65536,(100,3))
        torch.testing.assert_close(original.position_embedding(cells),centered.position_embedding(cells),atol=1e-6,rtol=1e-5)
        folded=torch.nn.Sequential(*[copy.deepcopy(layer) for layer in centered.octree_mlp.children()])
        with torch.no_grad():
            folded[0].bias.sub_(folded[0].weight@centered.position_input_center)
            folded[2].bias.sub_(folded[2].weight@centered.octree_mlp.hidden_center)
        torch.testing.assert_close(folded(positional_bits(cells,16)),centered.position_embedding(cells),atol=1e-6,rtol=1e-5)
        centered.position_embedding(cells).square().sum().backward()
        self.assertGreater(float(centered.octree_mlp[0].weight.grad.norm()),0.)
        self.assertGreater(float(centered.octree_mlp[2].weight.grad.norm()),0.)
        health=centered.octree_health(sample_size=None)
        self.assertGreater(health['octree_hidden_nonzero_point_fraction'],0.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt';save_checkpoint(path,centered,None,4)
            restored,_,_,_=load_checkpoint(path)
            torch.testing.assert_close(restored.position_embedding(cells),centered.position_embedding(cells),rtol=0,atol=0)
            centered.freeze_coordinates();centered.eval()
            expected=centered.attributes(torch.zeros(3),25)
            path=Path(directory)/'both_centers.zip';encode_model(centered,path,'cpu')
            decoded,_=decode_model(path)
            actual=decoded.attributes(torch.zeros(3),25)
            for key in ('xyz','opacity','color','rotation','scales'):
                torch.testing.assert_close(actual[key],expected[key],atol=2e-6,rtol=2e-6)

    def test_regional_embedding_intervention_preserves_outside_and_checks_order(self):
        model=self.make_model()
        cells,_=quantize_cells(model.xyz,model.bounds_min,model.bounds_max,16)
        original=model.position_embedding(cells).detach()
        table=original.clone();table[:6]=original[:6].mean(0)
        with fixed_embedding_table(model,cells,table):
            actual=model.position_embedding(cells)
            torch.testing.assert_close(actual[6:],original[6:],rtol=0,atol=0)
            torch.testing.assert_close(actual[:6],original[:6].mean(0).expand(6,-1),rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'cell order'):
                model.position_embedding(cells.roll(1,0))
        self.assertNotIn('position_embedding',model.__dict__)
        torch.testing.assert_close(model.position_embedding(cells),original,rtol=0,atol=0)

    def test_relu_homogeneous_initializer_preserves_function_rng_and_other_weights(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(72);original=SmolGS(tiny_config(),device=device)
            rng=torch.get_rng_state().clone()
            torch.manual_seed(72);scaled=SmolGS(tiny_config(octree_init_gain=8),device=device)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for key,value in original.state_dict().items():
                if key not in ('octree_mlp.0.weight','octree_mlp.0.bias','octree_mlp.2.weight','octree_mlp.2.bias','octree_mlp.4.weight'):
                    torch.testing.assert_close(value,scaled.state_dict()[key],rtol=0,atol=0)
            cells=torch.randint(0,65536,(128,3),device=device)
            torch.testing.assert_close(original.position_embedding(cells),scaled.position_embedding(cells),rtol=0,atol=0)
            scaled.position_embedding(cells).square().sum().backward()
            self.assertGreater(float(scaled.octree_mlp[0].weight.grad.norm()),0.)
            self.assertTrue(all(not isinstance(layer,torch.nn.LeakyReLU) for layer in scaled.modules()))

    def test_dead_branch_repair_preserves_function_rng_and_bias_optimizer_state(self):
        model=self.make_model()
        with torch.no_grad():
            model.octree_mlp[2].weight.zero_();model.octree_mlp[2].bias.fill_(-1.)
        opt=model.make_optimizer();cells,_=quantize_cells(model.xyz,model.bounds_min,model.bounds_max,16)
        model.position_embedding(cells).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
        expected=model.position_embedding(cells).detach().clone()
        bias=model.octree_mlp[4].bias;bias_state=copy.deepcopy(opt.state[bias])
        unchanged={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('octree_mlp.')}
        rng=torch.get_rng_state().clone();report=repair(model,opt,19)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.testing.assert_close(model.position_embedding(cells),expected,rtol=0,atol=0)
        self.assertGreater(report['after']['octree_hidden_nonzero_point_fraction'],0.)
        for key,value in bias_state.items():torch.testing.assert_close(opt.state[bias][key],value,rtol=0,atol=0)
        for layer in (model.octree_mlp[0],model.octree_mlp[2]):
            for param in layer.parameters():self.assertNotIn(param,opt.state)
        self.assertNotIn(model.octree_mlp[4].weight,opt.state)
        for key,value in unchanged.items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        model.position_embedding(cells).sum().backward()
        self.assertGreater(float(model.octree_mlp[4].weight.grad.norm()),0.)
        opt.step();opt.zero_grad(set_to_none=True)
        model.position_embedding(cells).sum().backward()
        self.assertGreater(float(model.octree_mlp[0].weight.grad.norm()),0.)
        with self.assertRaisesRegex(ValueError,'not dead'):
            repair(model,opt,19)

    def test_checkpoint_centered_repair_preserves_output_and_roundtrips_centers(self):
        model=self.make_model()
        with torch.no_grad():
            model.octree_mlp[2].weight.zero_();model.octree_mlp[2].bias.fill_(-1.)
        opt=model.make_optimizer();cells,_=quantize_cells(model.xyz,model.bounds_min,model.bounds_max,16)
        before=model.position_embedding(cells).detach().clone();rng=torch.get_rng_state().clone()
        repair(model,opt,19,'current_points')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.testing.assert_close(model.position_embedding(cells),before,rtol=0,atol=0)
        torch.testing.assert_close(model.position_inputs(cells).mean(0),torch.zeros(48),atol=1e-7,rtol=0)
        h=model.octree_mlp.hidden_input(model.octree_mlp[0](model.position_inputs(cells)))
        torch.testing.assert_close(h.mean(0),torch.zeros(model.config.octree_mlp_width),atol=1e-7,rtol=0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'repair.pt';save_checkpoint(path,model,opt,12)
            loaded,_,_,_=load_checkpoint(path,resume=True)
            self.assertEqual(loaded.config.octree_input_center_mode,'checkpoint')
            self.assertEqual(loaded.config.octree_hidden_center_mode,'checkpoint')
            torch.testing.assert_close(loaded.position_embedding(cells),before,rtol=0,atol=0)
            torch.testing.assert_close(loaded.position_input_center,model.position_input_center,rtol=0,atol=0)
            torch.testing.assert_close(loaded.octree_mlp.hidden_center,model.octree_mlp.hidden_center,rtol=0,atol=0)

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

    def test_size_aware_adc_splits_and_preserves_adam_survivors(self):
        model = self.make_model()
        model.scene_extent.fill_(10.)
        optimizer = model.make_optimizer()
        (model.xyz.square().sum()+model.features.square().sum()+model.log_scales.square().sum()).backward()
        optimizer.step()
        with torch.no_grad():
            model.log_scales.fill_(math.log(0.01))
            model.log_scales[2].fill_(math.log(0.4))  # large, but below prune size
            model.grad_accum[[1, 2]] = 1.
            model.vis_count.fill_(1.)
            model.opacity_accum.fill_(0.5)
        original_xyz = model.xyz.detach().clone()
        original_features = model.features.detach().clone()
        moments = optimizer.state[model.log_scales]['exp_avg'].clone()
        added, pruned = model.density_control(optimizer, torch.zeros(3), 6, max_splats=14)
        self.assertEqual((added, pruned, len(model.xyz)), (3, 1, 14))
        self.assertEqual(model.last_adc_report['cloned'], 1)
        self.assertEqual(model.last_adc_report['split_parents'], 1)
        # Order: eleven survivors, one exact-position small clone, two split children.
        torch.testing.assert_close(model.xyz[11], original_xyz[1])
        torch.testing.assert_close(model.features[12:], original_features[2].expand(2, -1))
        torch.testing.assert_close(model.log_scales[12:].exp(), torch.full((2, 3), 0.25))
        keep = torch.tensor([i for i in range(12) if i != 2])
        torch.testing.assert_close(optimizer.state[model.log_scales]['exp_avg'][:11], moments[keep])
        for parameter in (model.xyz, model.features, model.log_scales):
            self.assertEqual(float(optimizer.state[parameter]['exp_avg'][11:].abs().sum()), 0.)
            self.assertEqual(float(optimizer.state[parameter]['exp_avg_sq'][11:].abs().sum()), 0.)

    def test_size_pruning_uses_fixed_camera_extent_and_not_bbox(self):
        model = self.make_model()
        model.scene_extent.fill_(10.)
        with torch.no_grad():
            model.log_scales.fill_(math.log(0.01))
            model.log_scales[0].fill_(math.log(10.))
            model.xyz[0, 0] = 50000.
            model.vis_count.fill_(1.)
            model.opacity_accum.fill_(0.5)
            model.last_seen.fill_(4000.)  # far point is still visible and opaque
        added, pruned = model.density_control(model.make_optimizer(), torch.zeros(3), 4000)
        self.assertEqual((added, pruned), (0, 1))
        self.assertEqual(model.last_adc_report['pruned_oversized'], 1)
        self.assertLess(float(model.xyz.abs().max()), 10.)
        self.assertEqual(float(model.scene_extent), 10.)

    def test_size_aware_adc_respects_net_point_budget_on_cpu_and_cuda(self):
        devices = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model = self.make_model().to(device)
                model.scene_extent.fill_(10.)
                optimizer = model.make_optimizer()
                with torch.no_grad():
                    model.log_scales.fill_(math.log(0.01))
                    model.log_scales[0].fill_(math.log(0.4))
                    model.grad_accum[0] = 1.
                    model.vis_count.fill_(1.)
                    model.opacity_accum.fill_(0.5)
                # A split has two children but consumes one net slot.
                self.assertEqual(model.density_control(optimizer, torch.zeros(3, device=device),
                                                      6, max_splats=12), (0, 0))
                model.grad_accum[0] = 1.
                model.vis_count.fill_(1.)
                model.opacity_accum.fill_(0.5)
                self.assertEqual(model.density_control(optimizer, torch.zeros(3, device=device),
                                                      6, max_splats=13), (2, 1))
                self.assertEqual(len(model.xyz), 13)
                torch.testing.assert_close(model.log_scales[-2:].exp(),
                                           torch.full((2, 3), 0.25, device=device))

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

    def test_optional_physical_size_changes_only_size_pruning(self):
        model = self.make_model()
        model.scene_extent.fill_(10.)
        with torch.no_grad():
            model.log_scales.fill_(math.log(0.01))
            model.log_scales[0].fill_(math.log(10.))
            model.vis_count.fill_(1.)
            model.opacity_accum.fill_(0.5)
            model.last_seen.fill_(4000.)
        baseline = copy.deepcopy(model)
        self.assertEqual(baseline.density_control(None, torch.zeros(3), 4000), (0, 1))
        self.assertEqual(model.density_control(None, torch.zeros(3), 4000,
                                              physical_size=torch.full((12,), 0.01)), (0, 0))
        model.vis_count.fill_(1.)
        model.opacity_accum.fill_(0.5)
        physical = torch.full((12,), 0.01); physical[0] = float('nan')
        self.assertEqual(model.density_control(None, torch.zeros(3), 4000, physical_size=physical), (0, 1))

    def test_audit_multiview_scales_and_readonly_bounds(self):
        model = self.make_model()
        before_rng = torch.get_rng_state()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        mask = torch.zeros(12, dtype=torch.bool); mask[[1, 4, 7]] = True
        centers = torch.tensor([[0., 0., 0.], [1., 2., 3.], [-2., 0., 2.]])
        expected = torch.stack([model.attributes(center, 6, mask)['scales'].amax(-1)
                                for center in centers]).amax(0)
        actual = effective_candidate_sizes(model, mask, centers, batch_views=2)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        view = readonly_view(model, mask, refresh=True)
        self.assertEqual(len(view.xyz), 3)
        self.assertFalse(torch.equal(view.bounds_min, model.bounds_min))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        parameter = model.xyz
        with self.assertRaisesRegex(ValueError, 'physical_size'):
            model.density_control(None, torch.zeros(3), 4000, physical_size=torch.zeros(12, 3))
        self.assertIs(model.xyz, parameter)

    def test_legacy_checkpoint_readable_but_not_silently_resumed(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'legacy.pt'
            save_checkpoint(path, model, model.make_optimizer(), 4)
            payload = torch.load(path, weights_only=False)
            del payload['model']['scene_extent']
            torch.save(payload, path)
            restored, _, _, _ = load_checkpoint(path)
            torch.testing.assert_close(restored.xyz, model.xyz)
            with self.assertRaisesRegex(ValueError, 'Legacy ADC checkpoint'):
                load_checkpoint(path, resume=True)

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
            optimizer.step()
        self.assertEqual(len(stages), 5)
        model.eval()
        before = render(model, camera, 25, 'torch')['image']
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
                if codec == 'cuda':
                    model.config.hash_backend = 'cuda'
                with torch.no_grad():
                    for parameter in model.entropy_mlp.parameters():
                        parameter.zero_()  # Fixed steps: features=1, scales=0.001.
                    model.features[0, :2] = model.features.new_tensor([-2_000_000., 2_000_000.])
                    model.log_scales[0] = model.log_scales.new_tensor([0.001, 4000., 1.]).log()
                    model.freeze_coordinates()
                    model.eval()
                    _, _, _, _, qf, qs = model.prior(model.xyz)
                    expected_f = torch.round(model.features/qf)*qf
                    expected_s = (torch.round(model.log_scales.exp()/qs)*qs).clamp_min(1e-7)
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
                if codec == 'cuda':
                    model.config.hash_backend = 'cuda'
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
                    return (xyz.new_zeros(n, d), xyz.new_ones(n, d),
                            xyz.new_zeros(n, 3), xyz.new_ones(n, 3),
                            q.expand(n, 1), xyz.new_full((n, 1), 0.001))

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
    def test_hac_cuda_hash_matches_torch(self):
        try:
            import _gridencoder
        except ImportError:
            self.skipTest('HAC grid extension unavailable')
        for dimensions in [2, 3]:
            reference = HashGrid(dimensions, [6, 18], 6, 4, 'torch').cuda()
            accelerated = HashGrid(dimensions, [6, 18], 6, 4, 'cuda').cuda()
            accelerated.load_state_dict(reference.state_dict())
            points = torch.cat([torch.rand(19, dimensions), torch.zeros(1, dimensions), torch.ones(1, dimensions)]).cuda()
            actual = accelerated(points)
            expected = reference(points)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            actual.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(accelerated.table.grad, reference.table.grad, atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
