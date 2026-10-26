from pathlib import Path
import json
import time
import numpy as np
from PIL import Image
import torch
from utils.loss_utils import LPIPS, psnr, ssim
from gaussian_renderer import render


@torch.no_grad()
def evaluate(model, cameras, step, backend='cuda', output=None, use_lpips=True, save_images=False):
    if not cameras:
        raise ValueError('Cannot evaluate an empty split')
    was_training = model.training
    model.eval()
    rows = []
    print(f'[Eval] step={step}, views={len(cameras)}, output={output}', flush=True)
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if save_images:
            (output/'renders').mkdir(exist_ok=True)
            (output/'gt').mkdir(exist_ok=True)
    try:
        lpips_metric = LPIPS(model.xyz.device) if use_lpips else None
        # Warm up kernels so the first image does not dominate render latency.
        render(model, cameras[0].to(model.xyz.device), step, backend)
        for index, camera_info in enumerate(cameras):
            camera = camera_info.to(model.xyz.device)
            gt = camera_info.image(model.xyz.device)
            if model.xyz.is_cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            image = render(model, camera, step, backend)['image'].clamp(0, 1)
            if model.xyz.is_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            row = {'name': camera.name, 'psnr': float(psnr(image, gt)),
                   'ssim': float(ssim(image, gt)), 'render_seconds': elapsed}
            if lpips_metric:
                row['lpips'] = float(lpips_metric(image, gt))
            rows.append(row)
            if (index+1) % 10 == 0 or index+1 == len(cameras):
                print(f'[Eval] {index+1}/{len(cameras)} views '
                      f'({100*(index+1)/len(cameras):.1f}%) | '
                      f'mean PSNR={sum(r["psnr"] for r in rows)/len(rows):.3f}', flush=True)
            if output and save_images:
                for folder, value in [('renders', image), ('gt', gt)]:
                    pixels = (value.permute(1, 2, 0).cpu().numpy()*255).round().clip(0, 255).astype(np.uint8)
                    Image.fromarray(pixels).save(output/folder/f'{index:05d}.png')
        report = {'views': len(rows), 'splats': len(model.xyz), 'step': step, 'renderer': backend,
                  'psnr': sum(r['psnr'] for r in rows)/len(rows),
                  'ssim': sum(r['ssim'] for r in rows)/len(rows),
                  'fps': len(rows)/sum(r['render_seconds'] for r in rows), 'per_view': rows,
                  'lpips': sum(r['lpips'] for r in rows)/len(rows) if lpips_metric else None,
                  'lpips_status': LPIPS.status if lpips_metric else 'not measured',
                  'lpips_protocol': LPIPS.protocol if lpips_metric else None}
        if output:
            (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
    finally:
        model.train(was_training)
