#!/usr/bin/env python3
"""Compare decoded rendering with the quantized checkpoint (no ground truth needed)."""
from argparse import ArgumentParser
from pathlib import Path
import json
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.checkpoint import load_checkpoint
from smolgs.codec import decode_model
from scene.dataset_readers import load_scene
from gaussian_renderer import render


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--archive', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--renderer', choices=['cuda', 'torch'], default='cuda')
    p.add_argument('--views', type=int, default=3)
    p.add_argument('--image-max-abs', type=float, default=1/255)
    p.add_argument('--image-mean-abs', type=float, default=1e-6)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint, _, step, metadata = load_checkpoint(args.checkpoint, args.device)
    decoded, report = decode_model(args.archive, args.device)
    data = metadata['data']
    _, cameras = load_scene(data['source_path'], data['images'], data['resolution'],
                            data['holdout'], data['white_background'])
    if len(checkpoint.xyz) != len(decoded.xyz):
        raise AssertionError('Splat count changed during codec roundtrip')
    checkpoint.eval()
    decoded.eval()
    errors = []
    with torch.no_grad():
        for camera in cameras[:args.views]:
            camera = camera.to(args.device)
            before_package = render(checkpoint, camera, step, args.renderer)
            after_package = render(decoded, camera, step, args.renderer)
            # Rasterization has hard support/alpha cutoffs: tiny floating-point
            # changes can affect a few pixels. Check attributes separately and
            # constrain both worst-pixel and mean image error.
            attribute_errors = {}
            for name in ('xyz', 'color', 'opacity', 'scales', 'rotation'):
                a, b = before_package['attributes'][name], after_package['attributes'][name]
                torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
                attribute_errors[name] = float((a-b).abs().max())
            before, after = before_package['image'], after_package['image']
            difference = (before-after).abs()
            if not torch.isfinite(difference).all() or float(difference.max()) > args.image_max_abs or float(difference.mean()) > args.image_mean_abs:
                raise AssertionError(f'Render disagreement: max={float(difference.max())}, '
                                     f'mean={float(difference.mean())}')
            errors.append({'name': camera.name, 'max_abs': float(difference.max()),
                           'mean_abs': float(difference.mean()), 'attributes_max_abs': attribute_errors})
    report.update({'checkpoint_step': step, 'render_agreement': True, 'views': errors,
                   'criteria': {'attributes_atol': 2e-5, 'attributes_rtol': 2e-5,
                                'image_max_abs': args.image_max_abs, 'image_mean_abs': args.image_mean_abs,
                                'quantized_symbols': 'all stream SHA-256 checks passed during decode'},
                   'note': 'Codec consistency check only; this does not establish paper metric reproduction.'})
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
