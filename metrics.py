#!/usr/bin/env python3
"""3DGS-style evaluation of saved test renders and matching ground truth PNGs."""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from utils.loss_utils import LPIPS, psnr, ssim


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_paths', '--model-paths', nargs='+', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--skip-lpips', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    metric = None if args.skip_lpips else LPIPS(args.device)
    for model_path in args.model_paths:
        root = Path(model_path)
        results, per_view = {}, {}
        methods = [p for p in (root/'test').glob('*') if (p/'renders').is_dir()]
        if not methods:
            raise FileNotFoundError(f'No test/*/renders folders in {root}; use render.py --save-images first')
        for method in sorted(methods):
            rows = []
            for render_file in sorted((method/'renders').glob('*.png')):
                gt_file = method/'gt'/render_file.name
                if not gt_file.is_file():
                    raise FileNotFoundError(gt_file)
                def read(path):
                    with Image.open(path) as image:
                        array = np.array(image.convert('RGB'), dtype=np.float32)/255
                    return torch.from_numpy(array).permute(2, 0, 1).to(args.device)
                image, gt = read(render_file), read(gt_file)
                with torch.no_grad():
                    row = {'name': render_file.name, 'PSNR': float(psnr(image, gt)), 'SSIM': float(ssim(image, gt))}
                    row['LPIPS'] = float(metric(image, gt)) if metric else None
                rows.append(row)
            if not rows:
                raise ValueError(f'No images in {method}')
            results[method.name] = {key: sum(row[key] for row in rows)/len(rows)
                                    if rows[0][key] is not None else None for key in ('PSNR', 'SSIM', 'LPIPS')}
            per_view[method.name] = rows
        (root/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        (root/'per_view.json').write_text(json.dumps(per_view, indent=2)+'\n')
        (root/'evaluation_protocol.json').write_text(json.dumps({
            'lpips_protocol': LPIPS.protocol if metric else None,
            'lpips_status': LPIPS.status if metric else 'not measured',
            'image_protocol': 'saved RGB PNGs in [0,1]; per-image arithmetic mean',
        }, indent=2)+'\n')
        print(json.dumps({str(root): results}, indent=2))


if __name__ == '__main__':
    main()
