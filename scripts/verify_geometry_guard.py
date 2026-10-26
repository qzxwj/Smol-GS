#!/usr/bin/env python3
"""Read-only geometry audit of an existing pre-quantization checkpoint.

No optimizer steps, checkpoint rewrites, or pruning of the source experiment.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scene.dataset_readers import load_scene
from utils.checkpoint import load_checkpoint
from utils.graphics_utils import quantize_cells, morton_codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source-path', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    model, _, step, metadata = load_checkpoint(args.checkpoint)
    if bool(model.coordinate_frozen):
        raise ValueError('Use a checkpoint before coordinate freezing')
    data = metadata.get('data', {})
    train, _ = load_scene(args.source_path, data.get('images', 'images'), data.get('resolution', 1),
                          data.get('holdout', 8), data.get('white_background', False))
    model.set_scene_extent(torch.stack([camera.center for camera in train]))
    size = model.log_scales.detach().amax(-1).exp()
    oversized = size > model.config.adc_prune_scale_ratio*model.scene_extent
    retained_xyz = model.xyz.detach()[~oversized]
    report = {'checkpoint': str(Path(args.checkpoint).resolve()), 'step': step,
              'splats': len(model.xyz), 'before': model.geometry_summary(),
              'oversized_controller_count': int(oversized.sum()),
              'scale_only_prune_remaining_bbox': (retained_xyz.amax(0)-retained_xyz.amin(0)).tolist()
                  if len(retained_xyz) else None,
              'note': 'Scale-only hypothetical selection, not a repair of the checkpoint or a quality evaluation.'}
    if len(retained_xyz):
        lo = retained_xyz.amin(0)
        hi = torch.maximum(retained_xyz.amax(0), lo+1e-6)
        cells, _ = quantize_cells(retained_xyz, lo, hi, model.config.octree_depth)
        unique = len(torch.unique(morton_codes(cells, model.config.octree_depth)))
        report['scale_only_prune_quantization'] = {
            'before': len(retained_xyz), 'unique_cells': unique,
            'collision_fraction': 1-unique/len(retained_xyz)}
    original_parameter = model.xyz
    original_bounds = (model.bounds_min.clone(), model.bounds_max.clone())
    try:
        removed = model.freeze_coordinates()
        report.update({'guard_rejected': False, 'removed': removed})
    except RuntimeError as error:
        report.update({'guard_rejected': True, 'guard_message': str(error),
                       'topology_unchanged': model.xyz is original_parameter,
                       'bounds_unchanged': all(torch.equal(a, b) for a, b in
                           zip(original_bounds, (model.bounds_min, model.bounds_max))),
                       'coordinates_still_trainable': model.xyz.requires_grad})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
