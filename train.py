#!/usr/bin/env python3
"""Train all five Smol-GS stages on an SfM scene."""
from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import random
import os
import time
import numpy as np
import torch
from arguments import Config, add_model_optimization_arguments, config_from_args
from scene.gaussian_model import SmolGS
from scene.dataset_readers import load_scene, load_points
from gaussian_renderer import render
from utils.loss_utils import loss_terms, psnr
from utils.checkpoint import save_checkpoint, load_checkpoint
from smolgs.codec import encode_model
from smolgs.evaluation import evaluate
from smolgs.coordinate_selection import score_training_opacity


def parser():
    p = ArgumentParser(description=__doc__)
    p.add_argument('-s', '--source-path', '--source_path', required=True)
    p.add_argument('-m', '--model-path', '--model_path', required=True)
    p.add_argument('-i', '--images', default='images')
    p.add_argument('-r', '--resolution', type=int, default=-1)
    p.add_argument('-w', '--white-background', '--white_background', action='store_true')
    p.add_argument('--holdout', type=int, default=8)
    p.add_argument('--variant', choices=['small', 'base', 'large'], default='base')
    p.add_argument('--config')
    p.add_argument('--device', '--data_device', default='cuda')
    p.add_argument('--gpu', type=int, help='Set CUDA_VISIBLE_DEVICES to this physical GPU before initialization')
    p.add_argument('--eval', action='store_true', help='Compatibility flag; every eighth image is always held out')
    p.add_argument('--renderer', choices=['cuda', 'torch'], default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', '--start_checkpoint')
    p.add_argument('--stop-after', type=int, help='Stop early without rescaling paper stages')
    p.add_argument('--max-initial-points', type=int, help='Debug only; changes paper initialization')
    p.add_argument('--max-splats', type=int, help='Optional memory limit; changes paper ADC')
    p.add_argument('--checkpoint-interval', type=int, default=1000)
    p.add_argument('--test-iterations', '--test_iterations', nargs='*', type=int, default=[7000, 15000, 20000, 30000, 35000])
    p.add_argument('--eval-lpips', action='store_true', help='Use cached pretrained VGG LPIPS weights')
    p.add_argument('--codec', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--skip-export', action='store_true')
    p.add_argument('--log-interval', type=int, default=100)
    p.add_argument('--monitor-octree', action='store_true', help='Log positional MLP activity without consuming RNG')
    p.add_argument('--require-active-octree', action='store_true', help='Stop if an all-point check confirms a dead positional MLP')
    p.add_argument('--coordinate-selection', choices=['first', 'opacity'], default=None,
                   help='Collision representative: first point or training-view mean visible opacity')
    add_model_optimization_arguments(p)
    return p


def main(args=None):
    args = parser().parse_args(args)
    if args.gpu is not None:
        if args.gpu < 0:
            raise ValueError('Select an explicit nonnegative GPU id')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    if args.renderer == 'cuda' and not args.device.startswith('cuda'):
        raise ValueError('CUDA renderer requires --device cuda')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check GPU access or use --device cpu --renderer torch for tiny validation')
    if min(args.checkpoint_interval, args.log_interval) < 1:
        raise ValueError('Log/checkpoint intervals must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = Path(args.model_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_cameras, test_cameras = load_scene(args.source_path, args.images, args.resolution,
                                           args.holdout, args.white_background)
    if args.resume:
        model, optimizer, first_step, metadata = load_checkpoint(args.resume, args.device, resume=True)
        stored = metadata.get('data', {})
        requested = {'source_path': str(Path(args.source_path).resolve()), 'images': args.images,
                     'resolution': args.resolution, 'holdout': args.holdout,
                     'white_background': args.white_background}
        if stored and stored != requested:
            raise ValueError('Resume data settings differ from checkpoint')
        if args.max_splats is None:
            args.max_splats = metadata.get('max_splats')
        if args.max_initial_points is None:
            args.max_initial_points = metadata.get('max_initial_points')
        config = model.config
        if args.config_geometry_mode is not None and args.config_geometry_mode != config.geometry_mode:
            raise ValueError('Cannot change geometry mode when resuming; start a fresh run')
        for name in ('geometry_quantization', 'vq_start_iter', 'rate_constrain_iter', 'vq_scale_cbsize',
                     'vq_rot_cbsize', 'vq_scale_lmbda', 'vq_rot_lmbda', 'vq_cb_lr', 'vq_logits_lr',
                     'vq_patch_size', 'reactivate_codeword_period'):
            override = getattr(args, 'config_'+name)
            if override is not None and override != getattr(config, name):
                raise ValueError(f'Cannot change {name} when resuming; use the checkpoint configuration')
    else:
        config = config_from_args(args)
        points = load_points(args.source_path)
        if args.max_initial_points:
            if args.max_initial_points < 2:
                raise ValueError('max-initial-points must be >=2')
            points = points[np.random.permutation(len(points))[:args.max_initial_points]]
        model = SmolGS(config, device=args.device)
        model.initialize(points)
        model.set_scene_extent(torch.stack([camera.center for camera in train_cameras]))
        optimizer = model.make_optimizer()
        first_step = 0
    resume_metadata = metadata if args.resume else {}
    stored_selection = resume_metadata.get('coordinate_selection', 'first')
    if (bool(model.coordinate_frozen) and args.coordinate_selection is not None
            and args.coordinate_selection != stored_selection):
        raise ValueError('Cannot change coordinate selection after coordinates are frozen')
    args.coordinate_selection = args.coordinate_selection or stored_selection
    metadata = {**resume_metadata, 'data': {'source_path': str(Path(args.source_path).resolve()), 'images': args.images,
                        'resolution': args.resolution, 'holdout': args.holdout,
                        'white_background': args.white_background},
                'train_names': [c.name for c in train_cameras], 'test_names': [c.name for c in test_cameras],
                'renderer': args.renderer, 'adc_gradient_space': 'screen_xy',
                'adc_topology_policy': model.adc_policy['topology_policy'],
                'max_initial_points': args.max_initial_points,
                'max_splats': args.max_splats, 'seed': args.seed,
                'hash_backend': config.hash_backend,
                'hash_detach_coordinates': config.hash_detach_coordinates,
                'appearance_backend': config.appearance_backend,
                'opacity_activation': config.opacity_activation,
                'octree_backend': config.octree_backend,
                'geometry_mode': config.geometry_mode,
                'coordinate_selection': args.coordinate_selection,
                'geometry_quantization': config.geometry_quantization,
                'effective_learning_rates': config.optimizer_rates(),
                'implementation': 'independent-paper-reproduction'}
    (output/'config.json').write_text(json.dumps(config.to_dict(), indent=2)+'\n')
    # Familiar 3DGS filename, stored as JSON and never evaluated as Python code.
    (output/'cfg_args').write_text(json.dumps({**metadata['data'], **config.to_dict()}, indent=2)+'\n')
    (output/'run.json').write_text(json.dumps({**metadata, 'args': vars(args)}, indent=2)+'\n')
    stop = min(args.stop_after or config.iterations, config.iterations)
    if stop <= first_step:
        raise ValueError('stop-after must exceed the resumed iteration')
    print(json.dumps({'start': first_step, 'stop': stop, 'splats': len(model.xyz),
                      'train_views': len(train_cameras), 'test_views': len(test_cameras),
                      'device': str(model.xyz.device), 'renderer': args.renderer,
                      'adc_gradient_space': 'screen_xy',
                      'adc_topology_policy': model.adc_policy['topology_policy'],
                      **model.geometry_summary()}), flush=True)
    started = time.perf_counter()
    stage_before = None
    model.train()
    csv_path = output/'training.csv'
    csv_has_header = csv_path.exists() and csv_path.stat().st_size > 0
    with (output/'training.jsonl').open('a') as log, csv_path.open('a', newline='') as csv_log:
        csv_fields = ['step', 'loss', 'psnr', 'splats', 'stage', 'view', 'elapsed_seconds']
        csv_writer = csv.DictWriter(csv_log, fieldnames=csv_fields)
        if not csv_has_header:
            csv_writer.writeheader()
            csv_log.flush()
        for step in range(first_step+1, stop+1):
            stage = config.stage(step)
            if stage != stage_before:
                print(f'iteration {step}: {stage}', flush=True)
                stage_before = stage
            if step == config.compact_until+1:
                with torch.no_grad():
                    model.refresh_bounds()
                    model.hash_min.copy_(model.bounds_min)
                    model.hash_max.copy_(model.bounds_max)
            if step == config.feature_until+1:
                scores = None
                if args.coordinate_selection == 'opacity' and not bool(model.coordinate_frozen):
                    scores, seen = score_training_opacity(model, train_cameras, step-1, args.renderer)
                    receipt = {'method': 'mean(opacity * (radii > 0)) over all training views',
                               'step': step-1, 'train_names': [c.name for c in train_cameras],
                               'scores': scores.cpu(), 'visible_view_counts': seen.cpu()}
                    torch.save(receipt, output/'coordinate_opacity_scores.pt')
                removed = model.freeze_coordinates(optimizer, scores=scores)
                del scores
                print(f'Quantized coordinates; removed {removed} duplicate cells', flush=True)
            model.prepare_geometry_quantization(step)
            model.update_learning_rates(optimizer, step)
            camera = train_cameras[random.randrange(len(train_cameras))].to(args.device)
            target = camera.image(args.device)
            optimizer.zero_grad(set_to_none=True)
            if model.optimizer_vq is not None:
                model.optimizer_vq.zero_grad(set_to_none=True)
            package = render(model, camera, step, args.renderer)
            loss, terms = loss_terms(model, package, target, step)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at iteration {step}')
            loss.backward()
            with torch.no_grad():
                model.collect_statistics(package['attributes'], package['visible'], step,
                                         package['viewspace_points'])
            optimizer.step()
            model.step_geometry_quantizers(step)
            # Release the raster graph before densification reallocates splat arrays.
            row = None
            if step % args.log_interval == 0 or step == stop:
                row = {'step': step, 'stage': stage, 'view': camera.name, 'loss': float(loss.detach()),
                       'psnr': float(psnr(package['image'].detach(), target)), 'splats': len(model.xyz),
                       **{k: float(v.detach()) for k, v in terms.items()},
                       'elapsed_seconds': time.perf_counter()-started}
            del loss, terms, package, target
            with torch.no_grad():
                if config.warmup_until < step <= config.compact_until and step % config.adc_interval == 0:
                    added, pruned = model.density_control(optimizer, camera.center, step, args.max_splats)
                    if row is not None:
                        row.update({'added': added, 'pruned': pruned, 'splats': len(model.xyz),
                                    **model.last_adc_report})
                elif step <= config.feature_until and step % config.adc_interval == 0:
                    model.refresh_bounds()
            if row:
                row.update(model.geometry_summary())
                if config.geometry_quantization == 'ecvq':
                    row.update(vq_updates=int(model.vq_updates), vq_rate_constrained=bool(model.vq_rate_constrained),
                               vq_cb_lr=model.optimizer_vq.param_groups[0]['lr'],
                               vq_logits_lr=model.optimizer_vq.param_groups[1]['lr'])
                if args.monitor_octree or args.require_active_octree:
                    health = model.octree_health()
                    if args.require_active_octree and health['octree_embedding_range'] == 0:
                        health = model.octree_health(sample_size=None)
                        if health['octree_second_active_fraction'] == 0:
                            (output/'octree_failure.json').write_text(json.dumps({'step': step, **health}, indent=2)+'\n')
                            raise RuntimeError(f'Positional MLP is dead at step {step}; see octree_failure.json')
                    row.update(health)
                log.write(json.dumps(row)+'\n')
                log.flush()
                csv_writer.writerow({key: row[key] for key in csv_fields})
                csv_log.flush()
                elapsed = row['elapsed_seconds']
                completed = step-first_step
                eta = elapsed/max(completed, 1)*(stop-step)
                print(f'[Train {output.parent.name}/{output.name}] '
                      f'{step}/{stop} ({100*step/stop:.1f}%) | {stage} | '
                      f'loss={row["loss"]:.6f} PSNR={row["psnr"]:.3f} | '
                      f'splats={len(model.xyz):,} | {completed/max(elapsed, 1e-9):.2f} it/s | '
                      f'ETA={eta/60:.1f} min'
                      +(f' | oct_points={row["octree_hidden_nonzero_point_fraction"]:.3%} '
                        f'oct_range={row["octree_embedding_range"]:.4g}' if 'octree_embedding_range' in row else ''), flush=True)
                print(json.dumps(row), flush=True)
            if step in args.test_iterations:
                report = evaluate(model, test_cameras, step, args.renderer,
                                  output/f'evaluation/iteration_{step}', args.eval_lpips)
                print(json.dumps({k: v for k, v in report.items() if k != 'per_view'}), flush=True)
            if step % args.checkpoint_interval == 0 or step == stop:
                save_checkpoint(output/'checkpoint_latest.pt', model, optimizer, step, metadata)
                if step == stop or step in args.test_iterations:
                    save_checkpoint(output/f'checkpoint_{step}.pt', model, optimizer, step, metadata)
    summary = {'completed_step': stop, 'paper_iterations': config.iterations,
               'all_stages_completed': stop == config.iterations, 'training_seconds': time.perf_counter()-started,
               'splats': len(model.xyz)}
    if stop == config.iterations and not args.skip_export:
        summary['compression'] = encode_model(model, output/'model.smolgs.zip', args.codec)
        print(json.dumps({k: v for k, v in summary['compression'].items()
                          if not k.startswith('components')}), flush=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
