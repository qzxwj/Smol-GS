#!/usr/bin/env python3
"""Verify shared Inria LPIPS evaluation against the reference on saved PNGs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.loss_utils import LPIPS, psnr, ssim


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', required=True, help='Directory containing renders/ and gt/')
    p.add_argument('--reference-root', required=True, help='Local checkout containing Inria lpipsPyTorch')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import lpips
    metric = LPIPS(args.device)
    # Build the actual local Inria implementation, loading identical cached
    # linear weights directly to avoid any network request or cache mutation.
    reference = Path(args.reference_root).resolve()
    sys.path.append(str(reference))
    from lpipsPyTorch.modules.lpips import LPIPS as ReferenceLPIPS
    from lpipsPyTorch.modules.networks import get_network, LinLayers
    original = ReferenceLPIPS.__new__(ReferenceLPIPS)
    torch.nn.Module.__init__(original)
    original.net = get_network('vgg')
    original.lin = LinLayers(original.net.n_channels_list)
    weights = Path(lpips.__file__).parent/'weights/v0.1/vgg.pth'
    state = torch.load(weights, map_location='cpu', weights_only=True)
    original.lin.load_state_dict({k.replace('lin', '').replace('model.', ''): v for k, v in state.items()})
    original.to(args.device).eval()
    images = Path(args.images).resolve()
    rows = []
    checks = []
    started = time.perf_counter()
    def read(path):
        with Image.open(path) as im:
            return torch.from_numpy(np.array(im.convert('RGB'), dtype=np.float32)/255).permute(2, 0, 1)[None].to(args.device)
    with torch.no_grad():
        for index, render_file in enumerate(sorted((images/'renders').glob('*.png'))):
            image, gt = read(render_file), read(images/'gt'/render_file.name)
            direct = float(metric(image[0], gt[0]))
            # Verify the package substitution against the actual reference
            # implementation on real full-resolution images at both ends.
            if index == 0 or index == 31:
                expected = float(original(image, gt).mean())
                checks.append({'name': render_file.name, 'reference': expected,
                               'package_direct': direct, 'abs_delta': abs(expected-direct)})
                if abs(expected-direct) > 1e-6:
                    raise AssertionError('Reference LPIPS mismatch')
            rows.append({'name': render_file.name, 'psnr': float(psnr(image, gt)),
                         'ssim': float(ssim(image, gt)), 'lpips': direct,
                         'render_sha256': hashlib.sha256(render_file.read_bytes()).hexdigest()})
            print(f'{index+1}: Inria LPIPS={direct:.6f}', flush=True)
    if not rows:
        raise ValueError('No saved images')
    sources = [reference/'metrics.py', *sorted((reference/'lpipsPyTorch').rglob('*.py'))]
    report = {'images': str(images), 'views': len(rows), 'device': args.device,
              'image_protocol': 'identical saved 8-bit RGB PNGs, full resolution; per-view arithmetic mean',
              'means': {k: sum(r[k] for r in rows)/len(rows) for k in
                        ('psnr', 'ssim', 'lpips')},
              'lpips_protocol': LPIPS.protocol, 'lpips_status': LPIPS.status,
              'reference_checks': checks, 'seconds': time.perf_counter()-started,
              'reference_sources_sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
              'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
              'source': 'https://github.com/graphdeco-inria/gaussian-splatting/blob/main/metrics.py',
              'caveat': 'Inria protocol verified. Smol-GS author evaluation code is unreleased, so its exact protocol is unconfirmed. PNG and float-render metrics differ.',
              'per_view': rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_view','reference_sources_sha256')}, indent=2))


if __name__ == '__main__':
    main()
