"""ECVQ behavior, geometry integration and exact resume checks."""
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

import numpy as np
import torch
from torch.nn import functional as F

from arguments import Config
from gaussian_renderer import render
from scene.dataset_readers import Camera
from scene.gaussian_model import SmolGS
from smolgs.codec import encode_model
from smolgs.vq import ECVQ
from test_core import tiny_config
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.loss_utils import loss_terms


def config(**changes):
    values = dict(geometry_mode='explicit', geometry_quantization='ecvq',
                  vq_start_iter=13, rate_constrain_iter=17, vq_scale_cbsize=4, vq_rot_cbsize=4,
                  vq_patch_size=5, reactivate_codeword_period=2)
    values.update(changes)
    return tiny_config(**values)


def model(device='cpu', **changes):
    current = SmolGS(config(**changes), device=device)
    points = torch.tensor([[-.2, -.1, 2.], [.1, -.15, 2.2], [-.1, .2, 2.1], [.25, .1, 2.3]], device=device)
    current.initialize(points)
    current.set_scene_extent(torch.tensor([[-1., 0., 0.], [1., 0., 0.]], device=device))
    with torch.no_grad():
        current.log_scales.copy_(current.xyz.new_tensor([[.08, .12, .05], [.09, .11, .06],
                                                        [.07, .13, .04], [.10, .10, .07]]).log())
        current.rotations.copy_(current.xyz.new_tensor([[1., .1, .2, .3], [1.1, .2, .1, .4],
                                                       [.9, .3, .2, .1], [1.2, .4, .3, .2]]))
    return current


class ECVQBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)

    def devices(self):
        return ['cpu']+(['cuda'] if torch.cuda.is_available() else [])


    def test_selected_codewords_bits_and_gradients(self):
        candidate = ECVQ(3, 2, 3, 2.)
        with torch.no_grad():
            candidate.initialized.fill_(True)
            candidate.codebook.copy_(torch.tensor([[[0., 0., 0.], [3., 0., 0.]]]))
            candidate.logits.zero_()
        x = torch.tensor([[.1, 0., 0.], [2.9, 0., 0.], [.2, 0., 0.]])
        result = candidate(x, True)
        self.assertEqual(result['x_index'].flatten().tolist(), [0, 1, 0])
        torch.testing.assert_close(result['x_hat'], torch.tensor([[0., 0., 0.], [3., 0., 0.], [0., 0., 0.]]))
        self.assertAlmostEqual(float(result['bits']), 3.)
        (result['x_hat'].sum()+result['bits']).backward()
        torch.testing.assert_close(candidate.codebook.grad, torch.tensor([[[2., 2., 2.], [1., 1., 1.]]]))
        torch.testing.assert_close(candidate.logits.grad, torch.tensor([[-.5, .5]])/math.log(2))
        with torch.no_grad():
            candidate.codebook[0, 0].fill_(20.)
        cached = candidate(x, True, index_cache=result['x_index'])
        self.assertEqual(cached['x_index'].flatten().tolist(), [0, 1, 0])
        torch.testing.assert_close(cached['x_hat'][0], torch.full((3,), 20.))

    def test_l2_not_squared_and_rate_changes_assignment(self):
        candidate = ECVQ(3, 2, 3, 2., patch_size=1)
        with torch.no_grad():
            candidate.initialized.fill_(True)
            candidate.codebook.copy_(torch.tensor([[[1., 0., 0.], [2., 0., 0.]]]))
            candidate.logits.copy_(torch.tensor([[0., 2.]]))
        x = torch.zeros(1, 3)
        self.assertEqual(candidate(x, False)['x_index'].item(), 0)
        self.assertEqual(candidate(x, True)['x_index'].item(), 1)
        squared_cost = candidate.codebook[0].square().sum(-1)-F.log_softmax(candidate.logits[0], dim=-1)/math.log(2)/2
        self.assertEqual(squared_cost.argmin().item(), 0)

    def test_single_empty_repeated_quaternions_and_unused_codes(self):
        for device in self.devices():
            candidate = ECVQ(4, 8, 4, 256., patch_size=2).to(device)
            x = torch.tensor([[1., 0., 0., 0.]], device=device)
            candidate.kmeans_initialize(x)
            self.assertTrue(torch.isfinite(candidate.logits).all())
            self.assertTrue(torch.isfinite(candidate.codebook).all())
            for count in (0, 1, 5):
                values = x.expand(count, -1).clone().requires_grad_(True)
                actual = candidate(values, True)
                self.assertEqual(actual['x_index'].shape, (count, 1))
                self.assertEqual(actual['x_hat'].shape, (count, 4))
                loss = (actual['x_hat']-values).norm(dim=-1).sum()+actual['bits']
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(candidate.codebook.grad).all())
                self.assertTrue(torch.isfinite(candidate.logits.grad).all())
                candidate.zero_grad(set_to_none=True)
            self.assertEqual(candidate.reactivate_codeword(), 7)
            self.assertTrue(torch.isfinite(candidate.logits).all())
            before = copy.deepcopy(candidate.state_dict())
            with self.assertRaisesRegex(ValueError, 'nonempty finite'):
                candidate.kmeans_initialize(x[:0])
            for key, value in before.items():
                torch.testing.assert_close(candidate.state_dict()[key], value, rtol=0, atol=0)

    def test_extreme_logits_assign_identically_in_training_and_no_grad(self):
        candidate = ECVQ(3, 2, 3, 1.)
        with torch.no_grad():
            candidate.initialized.fill_(True)
            candidate.codebook.copy_(torch.tensor([[[1., 0., 0.], [100., 0., 0.]]]))
            candidate.logits.copy_(torch.tensor([[-1000., 0.]]))
        x = torch.tensor([[1., 0., 0.]])
        training = candidate(x, True)
        with torch.no_grad(): inference = candidate(x, True)
        for key in training:
            torch.testing.assert_close(training[key], inference[key], rtol=0, atol=0)


class ECVQIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(71)
        random.seed(71)
        np.random.seed(71)

    def devices(self):
        return ['cpu']+(['cuda'] if torch.cuda.is_available() else [])

    def test_rdo_refresh_schedule_evaluation_and_topology_cache(self):
        for device in self.devices():
            current = model(device, vq_start_iter=20001, rate_constrain_iter=25001,
                            iterations=35000, feature_until=30000, compact_until=20000,
                            densify_until=15000, warmup_until=500)
            current.prepare_geometry_quantization(20001)
            center = current.xyz.new_zeros(3)
            quantizer = current.quantizers['scale']
            with mock.patch.object(quantizer, 'forward', wraps=quantizer.forward) as forward:
                first = current.attributes(center, 20001)['geometry_indices']['scale'].clone()
                self.assertIsNone(forward.call_args.kwargs['index_cache'])
                with torch.no_grad():
                    quantizer.codebook.copy_(quantizer.codebook.roll(1, dims=1))
                for step in range(20002, 20011):
                    attrs = current.attributes(center, step)
                    self.assertIsNotNone(forward.call_args.kwargs['index_cache'])
                    torch.testing.assert_close(attrs['geometry_indices']['scale'], first, rtol=0, atol=0)
                current.attributes(center, 20011)
                self.assertIsNone(forward.call_args.kwargs['index_cache'])
                current.eval()
                fresh = current.attributes(center, 20012)['geometry_indices']['scale']
                self.assertIsNone(forward.call_args.kwargs['index_cache'])
                torch.testing.assert_close(fresh, quantizer(current.log_scales.exp())['x_index'])
            current.train()
            old = current.scale_index_cache.clone()
            indices = torch.tensor([2, 0, 2], device=device)
            current.replace_topology(indices, new_start=2)
            torch.testing.assert_close(current.scale_index_cache[:2], old[indices[:2]])
            self.assertEqual(current.scale_index_cache[-1].item(), -1)
            current.attributes(center, 20012)
            self.assertTrue((current.scale_index_cache >= 0).all())
            # Coordinate freezing sorts/deduplicates points, preserving cached identities.
            old_xyz = current.xyz.detach().clone()
            old = current.scale_index_cache.clone()
            current.freeze_coordinates()
            for xyz, cached in zip(current.xyz, current.scale_index_cache):
                nearest = (old_xyz-xyz).square().sum(-1).argmin()
                torch.testing.assert_close(cached, old[nearest])

    def test_missing_checkpoint_cache_is_rejected(self):
        current = model()
        optimizer = current.make_optimizer()
        self.update(current, optimizer, 13)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'missing_cache.pt'
            save_checkpoint(path, current, optimizer, 13)
            payload = torch.load(path, weights_only=False)
            del payload['model']['scale_index_cache']
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                load_checkpoint(path, resume=True)

    def update(self, current, optimizer, step, backend='torch'):
        if step == current.config.feature_until+1:
            current.freeze_coordinates(optimizer)
        current.prepare_geometry_quantization(step)
        current.update_learning_rates(optimizer, step)
        optimizer.zero_grad(set_to_none=True)
        current.optimizer_vq.zero_grad(set_to_none=True)
        camera = Camera('probe', 16, 16, 14., 14., torch.eye(4, device=current.xyz.device))
        package = render(current, camera, step, backend)
        loss, terms = loss_terms(current, package, torch.full_like(package['image'], .25), step)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in current.parameters() if p.grad is not None))
        optimizer.step()
        current.step_geometry_quantizers(step)
        return package, terms

    def test_config_and_separate_optimizer_parameter_coverage(self):
        invalid = [{'geometry_mode': 'neural'}, {'vq_start_iter': 12}, {'rate_constrain_iter': 12},
                   {'rate_constrain_iter': 26}, {'vq_scale_lmbda': 0.}, {'vq_patch_size': 0}, {'vq_rot_cbsize': 0}]
        for changes in invalid:
            with self.assertRaises(ValueError): config(**changes)
        current = model()
        optimizer = current.make_optimizer()
        base = {p for group in optimizer.param_groups for p in group['params']}
        vq = {p for group in current.optimizer_vq.param_groups for p in group['params']}
        self.assertFalse(base & vq)
        self.assertEqual(base | vq, set(current.parameters()))
        self.assertEqual([g['lr'] for g in current.optimizer_vq.param_groups], [.0002, .002])
        torch.manual_seed(65)
        baseline = SmolGS(tiny_config(geometry_mode='explicit'))
        rng = torch.get_rng_state().clone()
        torch.manual_seed(65)
        ecvq = SmolGS(config())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(ecvq.state_dict()[name], value, rtol=0, atol=0)

    def test_rdo_55n_geometry_weight_raw_quaternion_and_ste_gradients(self):
        current = model(vq_scale_cbsize=1, vq_rot_cbsize=2, vq_scale_lmbda=8., vq_rot_lmbda=4.)
        current.prepare_geometry_quantization(13)
        with torch.no_grad():
            current.quantizers['scale'].codebook.fill_(.2)
            current.quantizers['rot'].codebook.copy_(torch.tensor([[[1., 0., 0., 0.], [3., 0., 0., 0.]]]))
            current.quantizers['rot'].logits.zero_()
            current.rotations.fill_(0.)
            current.rotations[:, 0] = 2.8
        mask = torch.tensor([True, False, False, False])
        attrs = current.attributes(torch.zeros(3), 13, mask)
        self.assertEqual(attrs['geometry_indices']['rot'].item(), 1)  # Raw 2.8, not normalized 1.
        scales = current.log_scales[mask].exp()
        expected_vq = ((scales-.2).norm(dim=-1).sum()+torch.tensor(.2))/(55*4)
        torch.testing.assert_close(attrs['geometry_vq_loss'], expected_vq)
        torch.testing.assert_close(attrs['geometry_rate_loss'], torch.tensor(1./4/(55*4)))
        torch.testing.assert_close(attrs['rotation'], torch.tensor([[1., 0., 0., 0.]]))
        image = torch.zeros(3, 16, 16)
        attrs['rate'] = torch.tensor(0.)
        total, _ = loss_terms(current, {'attributes': attrs, 'image': image}, image, 13)
        torch.testing.assert_close(total, expected_vq+attrs['geometry_rate_loss'])
        # The same coefficient applies relative to a nonzero image loss.
        image = torch.full((3, 16, 16), .1, requires_grad=True)
        target = torch.full_like(image, .3)
        package = {'attributes': attrs, 'image': image}
        combined, _ = loss_terms(current, package, target, 13)
        without_geometry = {**attrs, 'geometry_vq_loss': scales.new_zeros(()),
                            'geometry_rate_loss': scales.new_zeros(())}
        baseline, _ = loss_terms(current, {'attributes': without_geometry, 'image': image}, target, 13)
        torch.testing.assert_close(combined-baseline, expected_vq+attrs['geometry_rate_loss'])
        torch.testing.assert_close(torch.autograd.grad(combined, image, retain_graph=True)[0],
                                   torch.autograd.grad(baseline, image)[0])
        geom = current.quantized_geometry(current.log_scales.exp(), current.rotations, 13)
        loss = geom['scales'].sum()+geom['raw_rotation'].sum()
        grads = torch.autograd.grad(loss, (current.log_scales, current.rotations,
                                          current.quantizers['scale'].codebook, current.quantizers['rot'].codebook),
                                    allow_unused=True, retain_graph=True)
        torch.testing.assert_close(grads[0], current.log_scales.exp())
        torch.testing.assert_close(grads[1], torch.ones_like(current.rotations))
        self.assertIsNone(grads[2]); self.assertIsNone(grads[3])
        grads = torch.autograd.grad(geom['geometry_vq_loss'],
                                    (current.log_scales, current.rotations, current.quantizers['scale'].codebook,
                                     current.quantizers['rot'].codebook))
        for grad in grads:
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0.)

    def test_initialization_switches_reactivation_and_empty_mask(self):
        for device in self.devices():
            current = model(device)
            optimizer = current.make_optimizer()
            before = copy.deepcopy(current.quantizers.state_dict())
            for step in (1, 8, 12):
                self.update(current, optimizer, step, 'cuda' if device == 'cuda' else 'torch')
            for name, value in before.items():
                torch.testing.assert_close(current.quantizers.state_dict()[name], value, rtol=0, atol=0)
            self.assertEqual(len(current.optimizer_vq.state), 0)
            self.assertEqual(int(current.vq_updates), 0)
            with mock.patch.object(current.quantizers['scale'], 'reactivate_codeword',
                                   wraps=current.quantizers['scale'].reactivate_codeword) as reactivate:
                for step in range(13, 20):
                    self.update(current, optimizer, step, 'cuda' if device == 'cuda' else 'torch')
                    self.assertEqual(bool(current.vq_rate_constrained), step >= 17)
                self.assertEqual(reactivate.call_count, 2)  # 14 and 16; not 18.
            self.assertEqual(int(current.vq_updates), 7)
            for count in (0, 1):
                mask = torch.arange(4, device=device) < count
                attrs = current.attributes(current.xyz.new_zeros(3), 19, mask)
                self.assertEqual(attrs['geometry_indices']['scale'].shape, (count, 1))
                self.assertTrue(torch.isfinite(attrs['geometry_vq_loss']))

    def test_unprojected_scale_codebook_and_reference_scheduler_milestone(self):
        current = model()
        current.make_optimizer()
        current.prepare_geometry_quantization(13)
        with torch.no_grad(): current.quantizers['scale'].codebook.fill_(-1.)
        # Like RDO, stage preparation and optimizer updates do not project.
        current.prepare_geometry_quantization(17)
        current.scheduler_vq.last_epoch = 11999
        current.scheduler_vq._step_count = 12000
        current.vq_updates.fill_(11999)
        current.step_geometry_quantizers(17)
        torch.testing.assert_close(current.quantizers['scale'].codebook,
                                   torch.full_like(current.quantizers['scale'].codebook, -1.))
        self.assertEqual(current.scheduler_vq.last_epoch, 12000)
        self.assertEqual(int(current.vq_updates), 12000)
        torch.testing.assert_close(torch.tensor([g['lr'] for g in current.optimizer_vq.param_groups]),
                                   torch.tensor([.00002, .0002]))

    def test_resume_across_every_stage_boundary_and_rng(self):
        for device in self.devices():
            for boundary in (12, 13, 16, 17, 18, 19):
                with self.subTest(device=device, boundary=boundary), tempfile.TemporaryDirectory() as directory:
                    current = model(device)
                    optimizer = current.make_optimizer()
                    backend = 'cuda' if device == 'cuda' else 'torch'
                    for step in range(1, boundary+1): self.update(current, optimizer, step, backend)
                    path = Path(directory)/'checkpoint.pt'
                    save_checkpoint(path, current, optimizer, boundary)
                    frozen = copy.deepcopy(current.state_dict())
                    opt_before = copy.deepcopy(current.optimizer_vq.state_dict())
                    scheduler_before = copy.deepcopy(current.scheduler_vq.state_dict())
                    self.update(current, optimizer, boundary+1, backend)
                    expected_rng = (random.random(), np.random.rand(), torch.rand(4),
                                    torch.rand(4, device=device))
                    restored, opt, step, _ = load_checkpoint(path, device, resume=True)
                    self.assertEqual(step, boundary)
                    for name, value in frozen.items():
                        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
                    self.assertEqual(restored.scheduler_vq.state_dict(), scheduler_before)
                    actual_opt = restored.optimizer_vq.state_dict()
                    self.assertEqual(actual_opt['param_groups'], opt_before['param_groups'])
                    for key, state in opt_before['state'].items():
                        for name, value in state.items():
                            torch.testing.assert_close(actual_opt['state'][key][name], value, rtol=0, atol=0)
                    with mock.patch.object(restored.quantizers['scale'], 'kmeans_initialize',
                                           wraps=restored.quantizers['scale'].kmeans_initialize) as initialize:
                        self.update(restored, opt, boundary+1, backend)
                        self.assertEqual(initialize.call_count, int(boundary == 12))
                    actual_rng = (random.random(), np.random.rand(), torch.rand(4), torch.rand(4, device=device))
                    self.assertEqual(actual_rng[:2], expected_rng[:2])
                    for actual, expected in zip(actual_rng[2:], expected_rng[2:]):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    tolerance = dict(rtol=2e-5, atol=2e-7) if device == 'cuda' else dict(rtol=0, atol=0)
                    for name, value in current.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[name], value, **tolerance)

    def test_native_backends_five_stages_and_inference_checkpoint(self):
        if not torch.cuda.is_available(): self.skipTest('CUDA required')
        current = model('cuda', appearance_backend='tcnn', octree_backend='tcnn',
                        octree_mlp_width=64, hash_backend='cuda')
        optimizer = current.make_optimizer()
        for step in range(1, 26): self.update(current, optimizer, step, 'cuda')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'native.pt'
            save_checkpoint(path, current, optimizer, 25)
            restored, _, step, _ = load_checkpoint(path, 'cuda')
            current.eval(); restored.eval()
            camera = Camera('probe', 16, 16, 14., 14., torch.eye(4, device='cuda'))
            with torch.no_grad():
                before = render(current, camera, step, 'cuda')
                after = render(restored, camera, step, 'cuda')
            torch.testing.assert_close(before['image'], after['image'], rtol=0, atol=0)
            encode_model(current, Path(directory)/'model.zip', 'cuda')

    def test_corrupted_checkpoint_training_states_are_rejected(self):
        current = model()
        optimizer = current.make_optimizer()
        self.update(current, optimizer, 13)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            save_checkpoint(path, current, optimizer, 13)
            original = torch.load(path, weights_only=False)
            for corruption in ('scheduler', 'optimizer', 'stage', 'format'):
                payload = copy.deepcopy(original)
                if corruption == 'scheduler': payload['scheduler_vq']['last_epoch'] += 1
                if corruption == 'optimizer': payload['optimizer_vq'] = None
                if corruption == 'stage': payload['model']['vq_rate_constrained'].fill_(True)
                if corruption == 'format': payload['format'] = 'smolgs-checkpoint-v4-explicit'
                torch.save(payload, path)
                with self.assertRaises(ValueError): load_checkpoint(path, resume=True)

    def test_training_entrypoint_resumes_both_optimizers_and_preserves_step_one(self):
        from train import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current_config = config(densify_threshold=1e6)
            config_path = root/'ecvq.json'
            config_path.write_text(json.dumps(current_config.to_dict()))
            cameras = [Camera('left', 16, 16, 14., 14., torch.eye(4)),
                       Camera('right', 16, 16, 14., 14., torch.eye(4))]
            cameras[1].world_to_camera[0, 3] = .1
            points = model().xyz.detach().numpy()
            common = ['-s', str(root/'data'), '--config', str(config_path), '--device', 'cpu',
                      '--renderer', 'torch', '--codec', 'cpu', '--test-iterations',
                      '--log-interval', '1', '--checkpoint-interval', '25']
            resumed_dir, continuous_dir = root/'resumed', root/'continuous'
            with mock.patch('train.load_scene', return_value=(cameras, cameras[:1])), \
                    mock.patch('train.load_points', side_effect=lambda _: points.copy()), \
                    mock.patch.object(Camera, 'image', return_value=torch.full((3, 16, 16), .25)), \
                    redirect_stdout(io.StringIO()):
                common.append('--skip-export')
                main([*common, '-m', str(resumed_dir), '--stop-after', '12'])
                checkpoint = resumed_dir/'checkpoint_latest.pt'
                with self.assertRaisesRegex(ValueError, 'Cannot change geometry_quantization'):
                    main([*common, '-m', str(resumed_dir), '--resume', str(checkpoint),
                          '--geometry-quantization', 'none'])
                main([*common, '-m', str(resumed_dir), '--resume', str(checkpoint), '--stop-after', '16'])
                main([*common, '-m', str(resumed_dir), '--resume', str(checkpoint)])
                main([*[arg for arg in common if arg != '--skip-export'], '-m', str(continuous_dir)])
            resumed, _, step, meta = load_checkpoint(resumed_dir/'checkpoint_latest.pt', resume=True)
            continuous, _, _, _ = load_checkpoint(continuous_dir/'checkpoint_latest.pt', resume=True)
            self.assertEqual(step, 25)
            self.assertEqual(meta['geometry_quantization'], 'ecvq')
            self.assertEqual(int(resumed.vq_updates), 13)
            self.assertEqual(resumed.scheduler_vq.state_dict(), continuous.scheduler_vq.state_dict())
            for name, value in continuous.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
            for key, state in continuous.optimizer_vq.state_dict()['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(resumed.optimizer_vq.state_dict()['state'][key][name], value, rtol=0, atol=0)
            self.assertFalse((resumed_dir/'model.smolgs.zip').exists())
            self.assertTrue((continuous_dir/'model.smolgs.zip').exists())
            rows = [json.loads(row) for row in (continuous_dir/'training.jsonl').read_text().splitlines()]
            self.assertEqual(rows[11]['geometry_vq_loss'], 0.)
            self.assertEqual(rows[12]['vq_updates'], 1)
            self.assertFalse(rows[15]['vq_rate_constrained'])
            self.assertTrue(rows[16]['vq_rate_constrained'])

    def test_topology_changes_preserve_survivors_and_initialize_new_indexes(self):
        current = model(vq_scale_cbsize=8, vq_rot_cbsize=8)
        optimizer = current.make_optimizer()
        self.update(current, optimizer, 13)
        before = current.attributes(torch.zeros(3), 13)
        index = torch.tensor([2, 0, 2])
        vq_state = copy.deepcopy(current.quantizers.state_dict())
        current.replace_topology(index, optimizer, new_start=2)
        after = current.attributes(torch.zeros(3), 13)
        for key in ('scale', 'rot'):
            torch.testing.assert_close(after['geometry_indices'][key], before['geometry_indices'][key][index])
        for name in ('scales', 'rotation'):
            torch.testing.assert_close(after[name], before[name][index], rtol=0, atol=0)
        for name, value in vq_state.items():
            torch.testing.assert_close(current.quantizers.state_dict()[name], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
