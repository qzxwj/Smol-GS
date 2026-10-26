#!/usr/bin/env python3
"""Compare measured decoded models with the local author CSV snapshot."""
from argparse import ArgumentParser
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'tandt': ('TanksAndTemples', ['truck', 'train']),
            'blending': ('DeepBlending', ['playroom', 'drjohnson'])}
METRICS = [('psnr', 'PSNR'), ('ssim', 'SSIM'), ('lpips', 'LPIPS'), ('size_MB', 'Size [Bytes]')]

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    report = json.loads((root/'benchmark_results.json').read_text())
    rows = []
    for measured in report['per_scene']:
        if measured['dataset'] not in DATASETS:
            continue
        dataset_name, _ = DATASETS[measured['dataset']]
        path = ROOT/'official_snapshot/results'/dataset_name/(measured['scene']+'.csv')
        official = next(row for row in csv.DictReader(path.open()) if row['Submethod'] == measured['variant'])
        reference = {metric: float(official[column])/(1e6 if metric == 'size_MB' else 1)
                     for metric, column in METRICS}
        rows.append(dict(dataset=measured['dataset'], scene=measured['scene'], variant=measured['variant'],
                         measured={metric: measured[metric] for metric, _ in METRICS}, paper=reference,
                         delta={metric: measured[metric]-reference[metric]
                                if measured[metric] is not None else None for metric, _ in METRICS}))
    for dataset, (_, scenes) in DATASETS.items():
        for variant in sorted({row['variant'] for row in rows if row['dataset'] == dataset}):
            group = [row for row in rows if row['dataset'] == dataset and row['variant'] == variant]
            if {row['scene'] for row in group} == set(scenes):
                average = {kind: {metric: sum(row[kind][metric] for row in group)/len(group)
                                 if all(row[kind][metric] is not None for row in group) else None
                                 for metric, _ in METRICS} for kind in ('measured', 'paper', 'delta')}
                rows.append(dict(dataset=dataset, scene='MEAN', variant=variant, **average))
    (root/'paper_comparison.json').write_text(json.dumps(rows, indent=2)+'\n')
    lines = ['# Smol-GS base: measured decoded model vs paper', '',
             'Delta = measured - paper. PSNR/SSIM: higher is better; LPIPS/size: lower is better.',
             'Paper reference: local official_snapshot CSV. Sizes include the entire archive in decimal MB.', '',
             '| Dataset | Scene | Metric | Measured | Paper | Delta |',
             '|---|---|---|---:|---:|---:|']
    for row in rows:
        for metric, _ in METRICS:
            values = [row[kind][metric] for kind in ('measured', 'paper', 'delta')]
            formatted = ['null' if value is None else f'{value:.6f}' for value in values]
            lines.append('| '+' | '.join([row['dataset'], row['scene'], metric]+formatted)+' |')
    text = '\n'.join(lines)+'\n'
    (root/'paper_comparison.md').write_text(text)
    print(text)

if __name__ == '__main__':
    main()
