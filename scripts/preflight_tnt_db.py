#!/usr/bin/env python3
"""Check the four benchmark scenes and exercise the CUDA training/codec pipeline."""
from argparse import ArgumentParser
from pathlib import Path
import importlib
import json
import os
import subprocess
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene.dataset_readers import load_scene, load_points
from utils.loss_utils import LPIPS

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    assert torch.cuda.is_available(), 'CUDA is required for the benchmark'
    for module in ('diff_gaussian_rasterization', 'simple_knn._C', '_gridencoder', 'tinycudann', 'arithmetic'):
        importlib.import_module(module)
    lpips = LPIPS('cuda')
    image = torch.zeros(3, 64, 64, device='cuda')
    assert abs(float(lpips(image, image))) < 1e-6
    del lpips, image
    torch.cuda.empty_cache()
    rows = []
    for dataset, scene in [('tandt', 'truck'), ('tandt', 'train'), ('blending', 'playroom'), ('blending', 'drjohnson')]:
        source = ROOT.parent/'data'/dataset/scene
        train, test = load_scene(source, 'images', 1, 8)
        assert not ({camera.name for camera in train} & {camera.name for camera in test})
        points = load_points(source)
        row = dict(dataset=dataset, scene=scene, train_views=len(train), test_views=len(test),
                   image_sizes=sorted({(camera.width, camera.height) for camera in train+test}),
                   sfm_points=len(points), holdout=8, resolution=1, images='images')
        rows.append(row)
        print(json.dumps(row), flush=True)
    output = Path(args.output)
    output.write_text(json.dumps(rows, indent=2)+'\n')
    smoke = output.parent/'validation/truck_cuda_50'
    command = [sys.executable, str(ROOT/'train.py'), '-s', str(ROOT.parent/'data/tandt/truck'),
               '-m', str(smoke), '-i', 'images', '-r', '160', '--config', str(ROOT/'configs/smoke.json'),
               '--max-initial-points', '2000', '--max-splats', '5000', '--test-iterations', '50',
               '--log-interval', '10', '--checkpoint-interval', '25', '--eval-lpips']
    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/verify_roundtrip.py'),
                    '--checkpoint', str(smoke/'checkpoint_50.pt'), '--archive', str(smoke/'model.smolgs.zip'),
                    '--output', str(smoke/'roundtrip.json')], cwd=ROOT, check=True)
    print('Preflight passed. The 50-step smoke result is only a pipeline check.', flush=True)

if __name__ == '__main__':
    main()
