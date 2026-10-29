"""Finish both prepared branches, independently decode, and validate all test views."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True)
    p.add_argument('--branches', nargs='+', choices=['global18', 'dual16'])
    p.add_argument('--skip-training', action='store_true', help='Validate already completed training/export')
    args = p.parse_args()
    experiment = Path(args.experiment).resolve()
    manifest = json.loads((experiment/'manifest.json').read_text())
    commands, results = [], {}
    python = sys.executable

    def run(argv, logfile):
        commands.append({'argv': argv, 'cwd': str(ROOT), 'log': str(logfile)})
        Path(str(logfile)+'.command.json').write_text(json.dumps(commands[-1], indent=2)+'\n')
        print(f'[Pipeline] {logfile.parent.name}: {logfile.name}', flush=True)
        with logfile.open('x') as log:
            subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    for name in args.branches or manifest['branches']:
        branch = experiment/name
        if not args.skip_training:
            run([python, 'train.py', '-s', manifest['data']['source_path'], '-m', str(branch),
                 '--resume', str(branch/'checkpoint_frozen_30000.pt'), '-r', '1',
                 '--test-iterations', '35000', '--eval-lpips', '--checkpoint-interval', '5000',
                 '--log-interval', '500', '--codec', 'cuda'], branch/'continuation.log')
        else:
            completed = json.loads((branch/'summary.json').read_text())
            assert completed['completed_step'] == 35000 and completed['all_stages_completed']
        archive = branch/'model.smolgs.zip'
        run([python, 'compress.py', 'decode', str(archive), str(branch/'decoded.pt'), '--device', 'cuda'],
            branch/'standalone_decode.log')
        run([python, 'scripts/verify_roundtrip.py', '--checkpoint', str(branch/'checkpoint_35000.pt'),
             '--archive', str(archive), '--output', str(branch/'roundtrip.json'), '--views', str(len(manifest['test_names']))],
            branch/'roundtrip.log')
        run([python, 'render.py', '-m', str(archive), '-s', manifest['data']['source_path'],
             '-r', '1', '--holdout', '8', '-o', str(branch/'decoded_evaluation')], branch/'decoded_evaluation.log')
        summary = json.loads((branch/'summary.json').read_text())
        metrics = json.loads((branch/'decoded_evaluation/metrics.json').read_text())
        roundtrip = json.loads((branch/'roundtrip.json').read_text())
        assert summary['completed_step'] == 35000 and summary['all_stages_completed']
        assert [r['name'] for r in metrics['per_view']] == manifest['test_names']
        assert metrics['views'] == len(manifest['test_names']) and roundtrip['render_agreement']
        assert len(roundtrip['views']) == len(manifest['test_names'])
        with zipfile.ZipFile(archive) as z:
            components = {}
            for info in z.infolist():
                group = info.filename.split('/')[0] if '/' in info.filename else info.filename
                components[group] = components.get(group, 0)+info.compress_size
        results[name] = {'immediate': manifest['immediate'][name], 'final_decoded':
                         {k: metrics[k] for k in ('psnr','ssim','lpips','views','splats')},
                         'bytes': archive.stat().st_size, 'size_MB': archive.stat().st_size/1e6,
                         'compressed_component_bytes': components,
                         'zip_overhead_bytes': archive.stat().st_size-sum(components.values()),
                         'archive': str(archive), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                         'roundtrip_all_views_passed': True,
                         'roundtrip_max_image_error': max(r['max_abs'] for r in roundtrip['views']),
                         'roundtrip_max_mean_image_error': max(r['mean_abs'] for r in roundtrip['views']),
                         'training_seconds': summary['training_seconds']}
        (branch/'acceptance.json').write_text(json.dumps(results[name], indent=2)+'\n')
        print(json.dumps({name: results[name]}, indent=2), flush=True)
    combined = {name: json.loads((experiment/name/'acceptance.json').read_text())
                for name in manifest['branches'] if (experiment/name/'acceptance.json').exists()}
    (experiment/'final_comparison.json').write_text(json.dumps(combined, indent=2)+'\n')


if __name__ == '__main__':
    main()
