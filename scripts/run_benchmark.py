#!/usr/bin/env python3
"""Sequential 13-scene benchmark, following the local HAC launch conventions."""
from argparse import ArgumentParser
from pathlib import Path
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCENES = {
    'mipnerf360': ['bicycle', 'garden', 'stump', 'room', 'counter', 'kitchen', 'bonsai', 'flowers', 'treehill'],
    'tandt': ['truck', 'train'], 'blending': ['playroom', 'drjohnson'],
}
INDOOR = {'room', 'counter', 'kitchen', 'bonsai'}


def main(dataset=None):
    p = ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['all']+list(SCENES), default=dataset or 'all')
    p.add_argument('--scene', nargs='+')
    p.add_argument('--data-root', '--data_root', default=str(ROOT.parent/'data'))
    p.add_argument('--output-root', '--output_root', default=str(ROOT/'outputs'))
    p.add_argument('--variant', choices=['small', 'base', 'large'], default='base')
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true', help='Resume each latest checkpoint, evaluate completed runs')
    p.add_argument('--log-interval', type=int, default=50)
    p.add_argument('--skip-lpips', action='store_true', help='Explicitly omit LPIPS when weights are unavailable')
    args = p.parse_args()
    datasets = list(SCENES) if args.dataset == 'all' else [args.dataset]
    unknown = set(args.scene or []) - {s for d in datasets for s in SCENES[d]}
    if unknown:
        p.error('Unknown scene(s): '+', '.join(sorted(unknown)))
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['OMP_NUM_THREADS'] = '8'
    env['PYTHONUNBUFFERED'] = '1'
    plan = []
    for dataset_name in datasets:
        for scene in SCENES[dataset_name]:
            if args.scene and scene not in args.scene:
                continue
            source = Path(args.data_root)/dataset_name/scene
            image_folder = ('images_2' if scene in INDOOR else 'images_4') if dataset_name == 'mipnerf360' else 'images'
            output = Path(args.output_root)/dataset_name/scene/args.variant
            train = [args.python, str(ROOT/'train.py'), '-s', str(source), '-m', str(output),
                     '-i', image_folder, '-r', '1', '--eval', '--variant', args.variant,
                     '--iterations', '35000', '--codec', 'cuda',
                     '--log-interval', str(args.log_interval)]
            evaluation = [args.python, str(ROOT/'render.py'), '-m', str(output/'model.smolgs.zip'),
                          '-s', str(source), '-i', image_folder, '-r', '1', '-o', str(output/'test/ours_35000'),
                          '--save-images']
            if args.skip_lpips:
                evaluation.append('--skip-lpips')
            else:
                train.append('--eval-lpips')
            plan.append((source, image_folder, output, train, evaluation))
    # Validate the entire queue before starting any long training job.
    errors = []
    for source, images, *_ in plan:
        if not (source/images).is_dir() or not (source/'sparse').is_dir():
            errors.append(str(source)+' (images or sparse missing)')
    if errors:
        p.error('Dataset preflight failed:\n'+'\n'.join(errors))
    for scene_index, (source, images, output, train, evaluation) in enumerate(plan, 1):
        print(f'\n[Scene {scene_index}/{len(plan)}] {source.parent.name}/{source.name} ({args.variant})', flush=True)
        verification = [args.python, str(ROOT/'scripts/verify_roundtrip.py'),
                        '--checkpoint', str(output/'checkpoint_latest.pt'),
                        '--archive', str(output/'model.smolgs.zip'),
                        '--output', str(output/'roundtrip.json')]
        commands = [train, verification, evaluation]
        latest, summary = output/'checkpoint_latest.pt', output/'summary.json'
        complete = False
        if args.resume and summary.is_file():
            complete = bool(json.loads(summary.read_text()).get('all_stages_completed'))
        checkpoint_info = Path(str(latest)+'.json')
        if args.resume and latest.is_file() and checkpoint_info.is_file():
            progress = json.loads(checkpoint_info.read_text())
            complete = complete or progress['step'] == progress['iterations'] == 35000
        if args.resume and latest.is_file() and not complete:
            train.extend(['--resume', str(latest)])
        if complete:
            if (output/'model.smolgs.zip').is_file():
                commands = [verification, evaluation]
            else:
                # A run may have saved its final checkpoint just before an export failure.
                export = [args.python, str(ROOT/'compress.py'), 'encode', str(latest),
                          str(output/'model.smolgs.zip')]
                commands = [export, verification, evaluation]
        for command in commands:
            print(f'CUDA_VISIBLE_DEVICES={args.gpu} '+shlex.join(command), flush=True)
            if not args.dry_run:
                output.mkdir(parents=True, exist_ok=True)
                with (output/'launcher.log').open('a') as log:
                    log.write('\n'+time.strftime('%Y-%m-%d %H:%M:%S')+' '+shlex.join(command)+'\n')
                    log.flush()
                    with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            print(line, end='', flush=True)
                        returncode = process.wait()
                    if returncode:
                        raise subprocess.CalledProcessError(returncode, command)
        if not args.dry_run:
            command = [args.python, str(ROOT/'scripts/collect_results.py'), '--output-root', args.output_root]
            subprocess.run(command, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
