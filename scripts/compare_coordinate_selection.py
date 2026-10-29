#!/usr/bin/env python3
"""Paired immediate evaluation of first-point versus training-view opacity selection.

Produces frozen checkpoints before evaluation (with original RNG/optimizer states)
for a subsequent matched continuation. Never updates parameters or reads GT for scoring.
"""
from argparse import ArgumentParser
import gc
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scene.dataset_readers import load_scene
from smolgs.coordinate_selection import score_training_opacity, select_coordinate_survivors
from smolgs.evaluation import evaluate
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.graphics_utils import quantize_cells, morton_codes


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compact(report):
    return {key: report[key] for key in ('psnr', 'ssim', 'lpips', 'views', 'splats')}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--renderer', choices=['cuda', 'torch'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory; existing experiment results are never overwritten')
    model, optimizer, step, meta = load_checkpoint(source, args.device, resume=True)
    if step != model.config.feature_until or bool(model.coordinate_frozen) or bool(model.decoded):
        raise ValueError('Use the unfrozen training checkpoint immediately before coordinate compression')
    data = meta['data']
    train, test = load_scene(data['source_path'], data['images'], data['resolution'], data['holdout'], data['white_background'])
    train_names, test_names = [c.name for c in train], [c.name for c in test]
    if (train_names != meta['train_names'] or test_names != meta['test_names']
            or set(train_names) & set(test_names)):
        raise ValueError('Training/test split does not match the source checkpoint')
    output.mkdir(parents=True)
    manifest = {'source_checkpoint': str(source), 'source_checkpoint_sha256': file_hash(source),
                'checkpoint_step': step, 'score_step': step, 'data': data,
                'train_names_used_for_scoring': train_names, 'test_names_used_only_for_evaluation': test_names,
                'method': 'sum(opacity * (radii > 0)) / number_of_training_views; pre-snap evaluation mode',
                'visibility': 'radii > 0 projection prefilter, not occlusion-aware contribution',
                'tie_break': 'lowest original point index', 'score_accumulation_dtype': 'float64',
                'decision_rule': 'continue if PSNR gain >= 0.05 dB, SSIM does not decrease and LPIPS does not increase',
                'selection_uses_test_metrics': False, 'continuation_decision_uses_test_metrics': True,
                'interpretation': 'Exploratory single-scene comparison; test metrics gate continuation, not score construction.',
                'source_hashes': {name: file_hash(ROOT/name) for name in (
                    'smolgs/coordinate_selection.py', 'scene/gaussian_model.py', 'gaussian_renderer/__init__.py',
                    'train.py', 'utils/checkpoint.py', 'scripts/compare_coordinate_selection.py')}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    before = model_hash(model)
    scores, seen = score_training_opacity(model, train, step, args.renderer)
    if model_hash(model) != before:
        raise RuntimeError('Scoring changed model state')
    with torch.no_grad():
        lo = model.xyz.amin(0)
        hi = torch.maximum(model.xyz.amax(0), lo+1e-6)
        cells, _ = quantize_cells(model.xyz, lo, hi, model.config.octree_depth)
        codes = morton_codes(cells, model.config.octree_depth)
        first = select_coordinate_survivors(codes)
        best = select_coordinate_survivors(codes, scores)
        assert torch.equal(codes[first], codes[best])
        assert bool((scores[best] >= scores[first]).all())
        _, counts = torch.unique(codes, return_counts=True)
        selection = {'input_points': len(codes), 'retained_points': len(best), 'collision_cells': int((counts > 1).sum()),
                     'changed_representatives': int((first != best).sum()), 'never_projected_points': int((seen == 0).sum()),
                     'state_unchanged_after_scoring': True}
    np.savez_compressed(output/'scores_and_indices.npz', scores=scores.cpu().numpy(),
                        visible_view_counts=seen.cpu().numpy(), first_indices=first.cpu().numpy(),
                        opacity_indices=best.cpu().numpy(), retained_morton_codes=codes[first].cpu().numpy())
    scores_cpu = scores.cpu()
    del model, optimizer, scores, seen, first, best, cells, codes
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    reports = {}
    for policy in ('first', 'opacity'):
        model, optimizer, source_step, source_meta = load_checkpoint(source, args.device, resume=True)
        branch = output/policy
        branch.mkdir()
        model.freeze_coordinates(optimizer, scores=scores_cpu.to(args.device) if policy == 'opacity' else None)
        assert len(model.xyz) == selection['retained_points']
        # Save BEFORE evaluate(): LPIPS construction can consume CPU RNG.
        prepared_meta = {**source_meta, 'coordinate_selection': policy, 'coordinate_selection_source': str(output/'manifest.json')}
        save_checkpoint(branch/'checkpoint_frozen_30000.pt', model, optimizer, source_step, prepared_meta)
        report = evaluate(model, test, source_step+1, args.renderer, branch/'immediate', use_lpips=True)
        if [r['name'] for r in report['per_view']] != test_names:
            raise RuntimeError('Evaluation view order changed between branches')
        reports[policy] = report
        del model, optimizer
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    baseline, candidate = reports['first'], reports['opacity']
    deltas = {key: candidate[key]-baseline[key] for key in ('psnr', 'ssim', 'lpips')}
    paired = [{'name': a['name'], **{key: b[key]-a[key] for key in ('psnr', 'ssim', 'lpips')}}
              for a, b in zip(baseline['per_view'], candidate['per_view'])]
    decision = deltas['psnr'] >= .05 and deltas['ssim'] >= 0 and deltas['lpips'] <= 0
    result = {'selection': selection, 'first': compact(baseline), 'opacity': compact(candidate),
              'delta_opacity_minus_first': deltas, 'paired_view_deltas': paired,
              'psnr_improved_views': sum(v['psnr'] > 0 for v in paired),
              'continue_5000_steps': decision, 'decision_rule': manifest['decision_rule']}
    (output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'summary.txt').write_text(json.dumps({k: v for k, v in result.items() if k != 'paired_view_deltas'}, indent=2)+'\n')
    print((output/'summary.txt').read_text(), flush=True)


if __name__ == '__main__':
    main()
