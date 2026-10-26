#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import importlib
import json
import platform
import subprocess
import sys
import torch


def main():
    p = ArgumentParser(description='Record interpreter, CUDA and extension availability')
    p.add_argument('--output')
    args = p.parse_args()
    report = {'python': sys.version, 'executable': sys.executable,
              'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
              'cuda_available': torch.cuda.is_available(), 'platform': platform.platform(),
              'extensions': {}}
    if torch.cuda.is_available():
        report['gpu'] = torch.cuda.get_device_name()
        report['compute_capability'] = list(torch.cuda.get_device_capability())
    for name in ['diff_gaussian_rasterization', 'simple_knn._C', '_gridencoder', 'arithmetic', 'lpips']:
        try:
            module = importlib.import_module(name)
            report['extensions'][name] = {'available': True, 'path': module.__file__}
        except (ImportError, OSError) as error:
            report['extensions'][name] = {'available': False, 'error': str(error)}
    print(json.dumps(report, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
