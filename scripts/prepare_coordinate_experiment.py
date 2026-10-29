"""Prepare matched 30k continuation checkpoints for global18 and dual16 coding."""
from argparse import ArgumentParser
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arguments import Config
from smolgs.coordinate_selection import score_training_opacity
from smolgs.evaluation import evaluate
from utils.checkpoint import load_checkpoint, save_checkpoint


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source = Path(args.checkpoint).resolve()
    model, _, step, meta = load_checkpoint(source, 'cuda', resume=True)
    assert step == model.config.feature_until and not bool(model.coordinate_frozen)
    from scene.dataset_readers import load_scene
    data = meta['data']
    train, test = load_scene(data['source_path'], data['images'], data['resolution'], data['holdout'], data['white_background'])
    assert [c.name for c in train] == meta['train_names'] and [c.name for c in test] == meta['test_names']
    scores, seen = score_training_opacity(model, train, step, 'cuda')
    torch.save({'scores': scores.cpu(), 'visible_views': seen.cpu(), 'train_names': meta['train_names']}, out/'scores.pt')
    scores = scores.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    manifest = {'checkpoint': str(source), 'checkpoint_sha256': digest(source),
                'selection': 'training-view mean opacity * projection visibility; no GT/test views',
                'branches': {'global18': {'depth': 18, 'tail': 0.}, 'dual16': {'depth': 16, 'tail': .001}},
                'continuation_steps': 5000, 'data': data,
                'test_names': meta['test_names'], 'immediate': {}, 'source_sha256': {}}
    for name in ('arguments/__init__.py', 'scene/gaussian_model.py', 'smolgs/coordinate_grid.py',
                 'smolgs/codec.py', 'utils/checkpoint.py', 'train.py', 'smolgs/coordinate_selection.py',
                 'scripts/prepare_coordinate_experiment.py', 'scripts/verify_roundtrip.py'):
        dest = out/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        manifest['source_sha256'][name] = digest(ROOT/name)
    for branch, settings in manifest['branches'].items():
        model, optimizer, step, meta = load_checkpoint(source, 'cuda', resume=True)
        model.config = Config(**{**model.config.to_dict(), 'coordinate_depth': settings['depth'],
                                  'coordinate_tail_fraction': settings['tail']})
        removed = model.freeze_coordinates(optimizer, scores.to('cuda'))
        branch_out = out/branch
        branch_out.mkdir()
        counts = torch.bincount(model.coordinate_regions, minlength=2).tolist()
        branch_meta = {**meta, 'coordinate_selection': 'opacity', 'coordinate_experiment': str(out),
                       'coordinate_branch': branch, 'source_checkpoint_sha256': manifest['checkpoint_sha256']}
        # Save continuation state before metric constructors consume RNG.
        save_checkpoint(branch_out/'checkpoint_frozen_30000.pt', model, optimizer, step, branch_meta)
        report = evaluate(model, test, step+1, 'cuda', branch_out/'immediate', use_lpips=True)
        manifest['immediate'][branch] = {k: report[k] for k in ('psnr','ssim','lpips','views','splats')}
        manifest['immediate'][branch].update(removed=removed, region_counts=counts,
            coordinate_lower=model.coordinate_bounds_min.tolist(), coordinate_upper=model.coordinate_bounds_max.tolist())
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    assert digest(source) == manifest['checkpoint_sha256']
    print(json.dumps(manifest['immediate'], indent=2), flush=True)


if __name__ == '__main__':
    main()
