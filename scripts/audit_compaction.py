#!/usr/bin/env python3
"""Replay Truck compaction and separate point deletion from octree-bound changes.

Diagnostics never modify weights or consume the training RNG. The optional
effective mode changes only the size criterion during compaction, measuring
predicted axes over ALL training cameras. No compression-stage training is run.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scene.gaussian_model import SmolGS
from scene.dataset_readers import load_scene
from gaussian_renderer import render
from utils.graphics_utils import positional_bits, quantize_cells
from utils.loss_utils import LPIPS, psnr, ssim
from utils.checkpoint import load_checkpoint, save_checkpoint, load_torch


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()


def same_rng(a, b):
    return (a[0] == b[0] and a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1])
            and a[1][2:] == b[1][2:] and torch.equal(a[2], b[2]) and torch.equal(a[3], b[3]))


@torch.no_grad()
def readonly_view(model, keep=None, refresh=False):
    """Independent topology/bounds, shared read-only MLPs; no model mutation."""
    result = copy.copy(model)
    result._parameters = model._parameters.copy()
    result._buffers = model._buffers.copy()
    for name in ('bounds_min', 'bounds_max'):
        result._buffers[name] = getattr(model, name).clone()
    if keep is not None:
        for name in ('xyz', 'features', 'log_scales'):
            result._parameters[name] = getattr(model, name)[keep]
        for name in ('cells', 'grad_accum', 'vis_count', 'opacity_accum', 'last_seen'):
            result._buffers[name] = getattr(model, name)[keep]
    if refresh:
        result.refresh_bounds()
    return result


@torch.no_grad()
def effective_candidate_sizes(model, candidates, camera_centers, batch_views=16):
    """Maximum final axis per candidate, across every provided camera."""
    indices = candidates.nonzero(as_tuple=True)[0]
    if not len(indices):
        return model.xyz.new_zeros(0)
    batch_views = min(batch_views, max(1, 250000//len(indices)))
    xyz = model.xyz[indices]
    cells, _ = quantize_cells(xyz, model.bounds_min, model.bounds_max, model.config.octree_depth)
    embedding = model.position_embedding(cells)
    features = model.features[indices]
    controllers = model.log_scales[indices].exp().clamp_min(1e-7)
    maxima = xyz.new_zeros(len(indices))
    for first in range(0, len(camera_centers), batch_views):
        centers = camera_centers[first:first+batch_views]
        direction = centers[:, None, :]-xyz[None]
        distance = direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        augmented = torch.cat([features[None].expand(len(centers), -1, -1), direction/distance,
                               distance, embedding[None].expand(len(centers), -1, -1)], dim=-1)
        predicted = torch.sigmoid(model.scale_mlp(augmented.reshape(-1, augmented.shape[-1])))
        physical = predicted.reshape(len(centers), len(indices), 3)*controllers[None]
        maxima = torch.maximum(maxima, physical.amax(dim=(0, 2)))
    return maxima


@torch.no_grad()
def reasons(model, step, centers):
    controller = model.log_scales.amax(-1).exp()
    finite = (torch.isfinite(model.xyz).all(-1) & torch.isfinite(model.log_scales).all(-1)
              & torch.isfinite(controller) & torch.isfinite(model.features).all(-1))
    opacity = model.opacity_accum/model.vis_count.clamp_min(1)
    low = (model.vis_count > 0) & (opacity < model.config.prune_opacity)
    invisible = (step-model.last_seen) >= model.config.invisible_patience
    limit = model.config.adc_prune_scale_ratio*model.scene_extent
    large = controller > limit
    physical = controller.clone()
    physical[large] = effective_candidate_sizes(model, large, centers)
    physical_large = physical > limit
    base = low | invisible | ~finite
    full = base | large
    corrected = base | physical_large
    assert not full.all() and not corrected.all(), 'Audit requires a nonempty representation'
    flags = low.to(torch.uint8)+2*invisible.to(torch.uint8)+4*large.to(torch.uint8)+8*(~finite).to(torch.uint8)
    return dict(controller=controller, physical=physical, opacity=opacity, low=low,
                invisible=invisible, finite=finite, large=large, base=base, full=full,
                corrected=corrected, flags=flags, limit=float(limit))


def quantiles(value):
    if not value.numel():
        return None
    return torch.quantile(value.float(), value.new_tensor([0., .5, .9, .99, 1.])).tolist()


@torch.no_grad()
def measure(model, cameras, step, metric=None):
    rows = []
    for camera_info in cameras:
        camera = camera_info.to('cuda')
        image = render(model, camera, step, 'cuda')['image'].clamp(0, 1)
        target = camera_info.image('cuda')
        row = {'name': camera.name, 'psnr': float(psnr(image, target)), 'ssim': float(ssim(image, target))}
        if metric is not None:
            row['lpips'] = float(metric(image, target))
        rows.append(row)
    return {'splats': len(model.xyz), 'views': len(rows),
            'lpips_protocol': LPIPS.protocol if metric is not None else None,
            'lpips_status': LPIPS.status if metric is not None else 'not measured',
            **{name: sum(row[name] for row in rows)/len(rows)
               for name in ('psnr', 'ssim', 'lpips') if name in rows[0]}, 'per_view': rows}


def cases(model, data):
    return {
        'A_before': readonly_view(model),
        'B_opacity_invisible_old_bounds': readonly_view(model, ~data['base']),
        'C_all_rules_old_bounds': readonly_view(model, ~data['full']),
        'D_all_rules_new_bounds': readonly_view(model, ~data['full'], True),
        'E_effective_size_new_bounds': readonly_view(model, ~data['corrected'], True),
        'F_opacity_invisible_new_bounds': readonly_view(model, ~data['base'], True),
    }


def full_counterfactual(path, output, centers, cameras, metric):
    model, _, step, _ = load_checkpoint(path, 'cuda')
    model.eval()
    with torch.no_grad():
        data = reasons(model, step, centers)
        results = {}
        for name, candidate in cases(model, data).items():
            results[name] = measure(candidate, cameras, step, metric)
            print(f'[Full counterfactual {step}] {name}: PSNR={results[name]["psnr"]:.4f} '
                  f'LPIPS={results[name]["lpips"]:.4f}', flush=True)
    write_json(output, {'step': step, 'cases': results,
                        'note': 'Same weights, no optimizer steps; only deletion masks and encoding bounds vary.'})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['baseline', 'effective', 'no_size'], default='baseline')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Construct LPIPS before train.main restores the checkpoint RNG.
    metric = LPIPS('cuda')
    train_cameras, test_cameras = load_scene('/root/autodl-tmp/data/tandt/truck', 'images', 1, 8, False)
    centers = torch.stack([camera.center for camera in train_cameras]).cuda()
    probes = [test_cameras[i] for i in (0, 8, 16, 24)]
    original = SmolGS.density_control
    records = []
    stable_ids = None
    worst_drop = float('inf')

    @torch.no_grad()
    def audited(model, optimizer, camera_center, step, max_splats=None, physical_size=None):
        nonlocal stable_ids, worst_drop
        assert model.config.lambda_opacity == 3e-7 and model.config.lambda_rate == 5e-4
        assert model.config.densify_until < step <= model.config.compact_until
        if stable_ids is None:
            stable_ids = torch.arange(len(model.xyz), device=model.xyz.device)
        before_rng = rng_state()
        started = time.perf_counter()
        data = reasons(model, step, centers)
        subset = data['large'] & ~data['base']
        rescued = subset & ~data['corrected']
        names = ('none', 'opacity', 'invisible', 'opacity+invisible', 'size', 'opacity+size',
                 'invisible+size', 'opacity+invisible+size')
        histogram = torch.bincount(data['flags'].long(), minlength=16).tolist()
        row = {'step': step, 'mode': args.mode, 'before': len(model.xyz),
               'size_limit': data['limit'], 'low_opacity': int(data['low'].sum()),
               'invisible': int(data['invisible'].sum()), 'oversized_controller': int(data['large'].sum()),
               'nonfinite': int((~data['finite']).sum()), 'size_only': int(subset.sum()),
               'size_only_all_training_views_below_limit': int(rescued.sum()),
               'size_candidates_all_training_views_below_limit': int((data['large'] &
                    (data['physical'] <= data['limit']) & data['finite']).sum()),
               'reason_histogram': {name: histogram[i] for i, name in enumerate(names)},
               'reason_histogram_all_codes': histogram,
               'size_only_controller_quantiles': quantiles(data['controller'][subset]),
               'size_only_effective_max_quantiles': quantiles(data['physical'][subset]),
               'size_only_opacity_quantiles': quantiles(data['opacity'][subset]),
               'old_bounds': [model.bounds_min.tolist(), model.bounds_max.tolist()],
               'training_cameras_for_physical_size': len(centers)}
        union = data['full']
        np.savez_compressed(output/f'candidates_{step}.npz',
                            point_ids=stable_ids[union].cpu().numpy(), flags=data['flags'][union].cpu().numpy(),
                            controller=data['controller'][union].cpu().numpy(),
                            effective_or_safe_upper_bound=data['physical'][union].cpu().numpy(),
                            opacity=data['opacity'][union].cpu().numpy(),
                            xyz=model.xyz[union].cpu().numpy())
        views = cases(model, data)
        proxy = {name: measure(candidate, probes, step) for name, candidate in views.items()}
        row['probe'] = {name: {k: v for k, v in report.items() if k != 'per_view'}
                        for name, report in proxy.items()}
        actual_case = {'baseline': 'D_all_rules_new_bounds', 'effective': 'E_effective_size_new_bounds',
                       'no_size': 'F_opacity_invisible_new_bounds'}[args.mode]
        drop = proxy[actual_case]['psnr']-proxy['A_before']['psnr']
        if drop < worst_drop:
            worst_drop = drop
            save_checkpoint(output/'worst_before_adc.pt', views['A_before'], None, step,
                            {'audit_state': 'before_adc; diagnostic only, no optimizer',
                             'data': {'source_path': '/root/autodl-tmp/data/tandt/truck', 'images': 'images',
                                      'resolution': 1, 'holdout': 8, 'white_background': False}})
            row['saved_as_worst'] = True
        if step == 20000:
            save_checkpoint(output/'final_before_adc.pt', views['A_before'], None, step,
                            {'audit_state': 'before_adc; diagnostic only, no optimizer'})
        assert same_rng(before_rng, rng_state()), 'Diagnostics consumed training RNG'
        criterion = (data['physical'] if args.mode == 'effective' else
                     torch.zeros_like(data['controller']) if args.mode == 'no_size' else None)
        result = original(model, optimizer, camera_center, step, max_splats, physical_size=criterion)
        actual_mask = (data['corrected'] if args.mode == 'effective' else
                       data['base'] if args.mode == 'no_size' else data['full'])
        assert result == (0, int(actual_mask.sum()))
        assert len(model.xyz) == row['before']-int(actual_mask.sum())
        stable_ids = stable_ids[~actual_mask]
        row.update({'after': len(model.xyz), 'actually_pruned': result[1],
                    'new_bounds': [model.bounds_min.tolist(), model.bounds_max.tolist()],
                    'audit_seconds': time.perf_counter()-started, 'rng_unchanged': True})
        records.append(row)
        with (output/'adc.jsonl').open('a') as log:
            log.write(json.dumps(row)+'\n')
        print(f'[ADC audit {args.mode}] {step}/20000 | N={row["before"]:,}->{row["after"]:,} | '
              f'opacity={row["low_opacity"]:,} invisible={row["invisible"]:,} '
              f'controller-large={row["oversized_controller"]:,} size-only={row["size_only"]:,} '
              f'size-only physically-small={row["size_only_all_training_views_below_limit"]:,} | '
              f'probe PSNR A/B/C/D/E/F='+ '/'.join(f'{proxy[name]["psnr"]:.3f}' for name in views), flush=True)
        return result

    manifest = {'mode': args.mode, 'parent_checkpoint': str(Path(args.checkpoint).resolve()),
                'parent_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                'lambda_opacity': 3e-7, 'lambda_rate': 5e-4, 'start': 15000, 'stop': 20000,
                'physical_cameras': len(centers), 'probe_names': [camera.name for camera in probes],
                'candidate_ids': 'Indices in the common 15k checkpoint, preserved across ADC events',
                'candidate_scale_measurement': 'Exact all-training-camera maximum is evaluated for controller-oversized candidates; other entries retain a safe controller upper bound.',
                'criterion': {'baseline': 'original controller maximum',
                              'effective': 'maximum final axis across all training cameras, compaction only',
                              'no_size': 'omit the extra size rule in compaction; opacity/invisibility/finite checks unchanged'}[args.mode],
                'cases': {'A': 'before deletion, old bounds', 'B': 'opacity/invisibility deletion, old bounds',
                          'C': 'all original rules deletion, old bounds',
                          'D': 'all original rules deletion, refreshed bounds',
                          'E': 'effective-size rule deletion, refreshed bounds',
                          'F': 'opacity/invisibility/finite deletion, refreshed bounds'}}
    write_json(output/'manifest.json', manifest)
    hashes = {}
    for source_name in ('scripts/audit_compaction.py', 'scene/gaussian_model.py', 'train.py'):
        source = Path(source_name)
        target = output/'runtime_source'/source_name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        target.write_bytes(content)
        hashes[source_name] = hashlib.sha256(content).hexdigest()
    write_json(output/'runtime_source_sha256.json', hashes)
    SmolGS.density_control = audited
    try:
        import train
        train.main(['-s', '/root/autodl-tmp/data/tandt/truck', '-m', str(output/'replay'),
                    '-i', 'images', '-r', '1', '--resume', args.checkpoint, '--stop-after', '20000',
                    '--log-interval', '50', '--checkpoint-interval', '1000', '--eval-lpips', '--skip-export'])
    finally:
        SmolGS.density_control = original
    assert len(records) == 50
    # Explicitly identify the diagnostic size criterion in this run's saved
    # metadata; model/Adam/RNG tensors and the original experiment stay intact.
    replay_run = output/'replay/run.json'
    run_data = json.loads(replay_run.read_text())
    run_data['compaction_audit'] = manifest
    write_json(replay_run, run_data)
    for path in (output/'replay').glob('checkpoint*.pt'):
        payload = load_torch(path)
        payload['metadata']['compaction_audit'] = manifest
        temporary = path.with_suffix('.metadata.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
    full = {}
    for name in ('worst', 'final'):
        full[name] = full_counterfactual(output/f'{name}_before_adc.pt', output/f'{name}_counterfactual.json',
                                         centers, test_cameras, metric)
    final_metrics = json.loads((output/'replay/evaluation/iteration_20000/metrics.json').read_text())
    summary = {'mode': args.mode, 'adc_events': len(records), 'lambda_opacity': 3e-7, 'lambda_rate': 5e-4,
               'low_opacity_total': sum(r['low_opacity'] for r in records),
               'invisible_total': sum(r['invisible'] for r in records),
               'controller_oversized_total': sum(r['oversized_controller'] for r in records),
               'size_only_total': sum(r['size_only'] for r in records),
               'size_only_physically_small_total': sum(r['size_only_all_training_views_below_limit'] for r in records),
               'actual_removed': sum(r['actually_pruned'] for r in records),
               'rng_preserved_at_all_events': all(r['rng_unchanged'] for r in records),
               'metrics_20000': {k: v for k, v in final_metrics.items() if k != 'per_view'},
               'worst_probe_step': min(records, key=lambda r: r['probe'][{
                   'baseline': 'D_all_rules_new_bounds', 'effective': 'E_effective_size_new_bounds',
                   'no_size': 'F_opacity_invisible_new_bounds'}[args.mode]]['psnr']-
                   r['probe']['A_before']['psnr'])['step'],
               'note': 'Counterfactuals are immediate same-weight comparisons, not retrained results.'}
    write_json(output/'audit_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
