#!/usr/bin/env python3
"""Collect measured metrics; never substitute paper numbers for experiment results."""
from argparse import ArgumentParser
from pathlib import Path
import csv
import json


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--output-root', required=True)
    args = p.parse_args()
    root = Path(args.output_root)
    rows = []
    for path in sorted(root.glob('*/*/*/test/ours_35000/metrics.json')):
        dataset, scene, variant = path.relative_to(root).parts[:3]
        report = json.loads(path.read_text())
        archive = root/dataset/scene/variant/'model.smolgs.zip'
        rows.append({'dataset': dataset, 'scene': scene, 'variant': variant,
                     'lpips_protocol': report.get('lpips_protocol') or report.get('lpips_status', 'legacy-unspecified'),
                     'psnr': report['psnr'], 'ssim': report['ssim'], 'lpips': report['lpips'],
                     'size_MB': archive.stat().st_size/1e6 if archive.is_file() else None,
                     'fps': report['fps'], 'splats': report['splats'], 'views': report['views']})
    averages = []
    for key in sorted({(r['dataset'], r['variant']) for r in rows}):
        group = [r for r in rows if (r['dataset'], r['variant']) == key]
        mean = {'dataset': key[0], 'variant': key[1], 'scenes_measured': len(group)}
        for metric in ['psnr', 'ssim', 'lpips', 'size_MB', 'fps']:
            values = [r[metric] for r in group]
            mean[metric] = sum(values)/len(values) if all(v is not None for v in values) else None
        protocols = sorted({r['lpips_protocol'] for r in group if r['lpips'] is not None})
        mean['lpips_protocols'] = protocols
        if len(protocols) > 1:
            mean['lpips'] = None
            mean['lpips_status'] = 'mixed protocols; no comparable average'
        averages.append(mean)
    root.mkdir(parents=True, exist_ok=True)
    report = {'per_scene': rows, 'dataset_averages': averages,
              'note': 'Means cover only scenes_measured; LPIPS null means not measured or mixed protocols. Legacy results retain their original protocol labels.'}
    (root/'benchmark_results.json').write_text(json.dumps(report, indent=2)+'\n')
    with (root/'benchmark_results.csv').open('w', newline='') as output:
        columns = ['dataset', 'scene', 'variant', 'psnr', 'ssim', 'lpips', 'size_MB', 'fps', 'splats', 'views', 'lpips_protocol']
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
