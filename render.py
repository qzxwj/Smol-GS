#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import json
import torch
from smolgs.codec import decode_model
from utils.checkpoint import load_checkpoint
from scene.dataset_readers import load_scene
from smolgs.evaluation import evaluate


def main():
    p = ArgumentParser(description='Render and evaluate a checkpoint or a standalone compressed archive')
    p.add_argument('-m', '--model', required=True, help='Checkpoint .pt or model.smolgs.zip')
    p.add_argument('-s', '--source-path', '--source_path', help='Scene path (required for archives)')
    p.add_argument('-o', '--output', required=True)
    p.add_argument('-i', '--images', default=None)
    p.add_argument('-r', '--resolution', type=int, default=None)
    p.add_argument('--holdout', type=int, default=None)
    p.add_argument('--white-background', action='store_true', default=None)
    p.add_argument('--split', choices=['train', 'test'], default='test')
    p.add_argument('--device', default='cuda')
    p.add_argument('--renderer', choices=['cuda', 'torch'], default='cuda')
    p.add_argument('--skip-lpips', action='store_true')
    p.add_argument('--save-images', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.model.endswith('.zip'):
        model, decode_report = decode_model(args.model, args.device)
        step, meta = model.config.iterations, {}
    else:
        model, _, step, meta = load_checkpoint(args.model, args.device)
        decode_report = None
    data = meta.get('data', {})
    source = args.source_path or data.get('source_path')
    if source is None:
        p.error('--source-path is required for a compressed archive')
    images = args.images if args.images is not None else data.get('images', 'images')
    resolution = args.resolution if args.resolution is not None else data.get('resolution', -1)
    holdout = args.holdout if args.holdout is not None else data.get('holdout', 8)
    white = args.white_background if args.white_background is not None else data.get('white_background', False)
    train, test = load_scene(source, images, resolution, holdout, white)
    report = evaluate(model, test if args.split == 'test' else train, step, args.renderer,
                      args.output, not args.skip_lpips, args.save_images)
    report['decode'] = decode_report
    report['source_model'] = str(Path(args.model).resolve())
    Path(args.output, 'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'per_view'}, indent=2))


if __name__ == '__main__':
    main()
