"""Paired grid-depth diagnostic; retain all points and original positional encoding.

Geometry-only cases hold every rendered attribute except xyz fixed. Full cases
recompute view direction and hash-conditioned feature quantization at snapped xyz.
Neither case changes the trained depth-16 positional network or deletes points.
"""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gaussian_renderer import render
from scene.dataset_readers import load_scene
from utils.checkpoint import load_checkpoint
from utils.graphics_utils import morton_codes, quantize_cells
from utils.loss_utils import LPIPS, psnr, ssim


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model, _, step, meta = load_checkpoint(args.checkpoint, 'cuda')
    assert step == model.config.feature_until and not bool(model.coordinate_frozen)
    model.eval()
    data = meta['data']
    _, cameras = load_scene(data['source_path'], data['images'], data['resolution'],
                            data['holdout'], data['white_background'])
    assert [c.name for c in cameras] == meta['test_names']
    xyz = model.xyz.clone()
    lo, hi = xyz.amin(0), xyz.amax(0)
    original_cells, _ = quantize_cells(xyz, model.bounds_min, model.bounds_max,
                                       model.config.octree_depth)
    result = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': digest(args.checkpoint),
        'script_sha256': digest(__file__), 'step': step, 'points': len(xyz),
        'method': __doc__, 'views': [c.name for c in cameras],
        'bbox_min': lo.tolist(), 'bbox_max': hi.tolist(), 'bbox_extent': (hi-lo).tolist(),
        'bounds_match_checkpoint': bool(torch.equal(lo, model.bounds_min) and torch.equal(hi, model.bounds_max)),
        'depth_changes_only_coordinate_grid': True,
        'training_updates': 0, 'points_deleted': 0,
        'region_statistics': [], 'grids': {}, 'metrics': {},
    }
    for tail in (.0001, .001, .005, .01):
        bounds = torch.quantile(xyz, xyz.new_tensor([tail, 1-tail]), dim=0)
        inside = ((xyz >= bounds[0]) & (xyz <= bounds[1])).all(-1)
        extent = bounds[1]-bounds[0]
        result['region_statistics'].append({
            'axis_tail_fraction': tail, 'lower': bounds[0].tolist(), 'upper': bounds[1].tolist(),
            'extent': extent.tolist(), 'inside_count': int(inside.sum()),
            'outside_count': int((~inside).sum()), 'outside_fraction': float((~inside).float().mean()),
            'depth16_grid_step': (extent/(1 << 16)).tolist(),
            'precision_gain_per_axis': ((hi-lo)/extent).tolist(),
        })
    centers = {}
    for depth in (16, 17, 18):
        cells, centers[depth] = quantize_cells(xyz, lo, hi, depth)
        _, counts = torch.unique(morton_codes(cells, depth), return_counts=True)
        movement = (centers[depth]-xyz).norm(dim=-1)
        result['grids'][str(depth)] = {
            'step': ((hi-lo)/(1 << depth)).tolist(),
            'displacement_p50_p90_p99_max': torch.quantile(movement, movement.new_tensor([.5,.9,.99,1.])).tolist(),
            'occupied_cells': len(counts), 'collision_cells': int((counts > 1).sum()),
            'would_delete_with_one_per_cell': int((counts-1).sum()),
        }
    names = ['baseline']+[f'{kind}_depth{d}' for d in (16,17,18) for kind in ('geometry_only','full')]
    rows = {name: [] for name in names}
    metric = LPIPS('cuda')
    model.cells = original_cells
    model.coordinate_frozen.fill_(True)
    for number, info in enumerate(cameras, 1):
        camera, target = info.to('cuda'), info.image('cuda')
        model.xyz.copy_(xyz)
        baseline = model.attributes(camera.center, step)
        for name in names:
            if name == 'baseline':
                attrs = baseline
            else:
                depth = int(name.rsplit('depth', 1)[1])
                if name.startswith('geometry_only'):
                    attrs = {**baseline, 'xyz': centers[depth]}
                else:
                    model.xyz.copy_(centers[depth])
                    attrs = model.attributes(camera.center, step)
                    assert torch.equal(attrs['scales'], baseline['scales'])
                    assert torch.equal(attrs['rotation'], baseline['rotation'])
            proxy = SimpleNamespace(xyz=attrs['xyz'], attributes=lambda *a, **kw: attrs)
            image = render(proxy, camera, step, 'cuda')['image'].clamp(0,1)
            rows[name].append({'name': info.name, 'psnr': float(psnr(image,target)),
                               'ssim': float(ssim(image,target)), 'lpips': float(metric(image,target))})
        if number % 5 == 0 or number == len(cameras):
            print(f'[Grid diagnostic] {number}/{len(cameras)} views', flush=True)
    for name in names:
        result['metrics'][name] = {key: sum(r[key] for r in rows[name])/len(rows[name])
                                   for key in ('psnr','ssim','lpips')}
        result['metrics'][name]['per_view'] = rows[name]
    for name in names[1:]:
        result['metrics'][name]['psnr_delta_from_baseline'] = result['metrics'][name]['psnr']-result['metrics']['baseline']['psnr']
    (out/'findings.json').write_text(json.dumps(result, indent=2)+'\n')
    compact = {k: {a:b for a,b in v.items() if a != 'per_view'} for k,v in result['metrics'].items()}
    (out/'summary.json').write_text(json.dumps({**result, 'metrics': compact}, indent=2)+'\n')
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == '__main__':
    main()
