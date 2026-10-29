"""Check complete per-step JSONL and CSV metrics for a finished fresh training run."""
from argparse import ArgumentParser
import csv
import json
import math
from pathlib import Path


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = Path(args.model)
    config = json.loads((model/'config.json').read_text())
    run = json.loads((model/'run.json').read_text())
    assert run['args']['resume'] is None, 'Expected fresh training'
    assert run['args']['log_interval'] == 1
    views = set(run['train_names'])
    count = 0
    with (model/'training.jsonl').open() as source, (model/'training.csv').open(newline='') as csv_source:
        reader = csv.DictReader(csv_source)
        for count, line in enumerate(source, 1):
            row = json.loads(line)
            assert row['step'] == count, f'Missing/repeated step at {count}'
            assert row['view'] in views
            assert all(math.isfinite(row[key]) for key in ('loss','psnr','elapsed_seconds'))
            assert isinstance(row['splats'], int) and row['splats'] > 0
            compact = next(reader)
            assert int(compact['step']) == count and int(compact['splats']) == row['splats']
            assert compact['view'] == row['view'] and compact['stage'] == row['stage']
            assert all(float(compact[key]) == row[key] for key in ('loss','psnr','elapsed_seconds'))
        assert next(reader, None) is None, 'Extra CSV rows'
    assert count == config['iterations'] == 35000, f'Incomplete log: {count}'
    summary = json.loads((model/'summary.json').read_text())
    assert summary['all_stages_completed'] and summary['completed_step'] == count
    assert summary['splats'] == row['splats']
    report = {'passed': True, 'fresh_training': True, 'rows': count, 'first_step': 1, 'last_step': count,
              'jsonl_csv_match': True, 'psnr_definition': 'current sampled training view, before optimizer update',
              'splats_definition': 'total Gaussians after the current step topology update',
              'final_training_view': row['view'], 'final_loss': row['loss'],
              'final_training_psnr': row['psnr'], 'final_splats': row['splats']}
    (model/'training_log_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
