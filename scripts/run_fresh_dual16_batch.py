"""Persistent four-scene, fresh dual16 training queue with per-step metric audits."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise ValueError('Use a new batch output directory')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['PYTHONUNBUFFERED'] = '1'
    import torch
    from arguments import Config
    from scene.dataset_readers import load_scene, load_points
    from utils.loss_utils import LPIPS
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    for module in ('diff_gaussian_rasterization', 'simple_knn._C', '_gridencoder', 'tinycudann', 'arithmetic', 'smolgs_rdo_entropy'):
        importlib.import_module(module)
    config_path = Path(args.config).resolve()
    config = Config.load(config_path)
    assert config.coordinate_depth == 16 and config.coordinate_tail_fraction == .001
    assert config.geometry_mode == 'explicit' and config.geometry_quantization == 'ecvq'
    metric = LPIPS('cuda')
    probe = torch.zeros(3,64,64,device='cuda')
    assert abs(float(metric(probe, probe))) < 1e-6
    del metric, probe
    torch.cuda.empty_cache()
    scenes = []
    for dataset, scene in [('tandt','truck'),('tandt','train'),('blending','playroom'),('blending','drjohnson')]:
        source = ROOT.parent/'data'/dataset/scene
        train, test = load_scene(source, 'images', 1, 8, False)
        points = load_points(source)
        assert len(points) >= 2 and len(train) > 0 and len(test) > 0
        assert not (set(c.name for c in train) & set(c.name for c in test))
        scenes.append({'dataset': dataset, 'scene': scene, 'source': str(source),
                       'output': str(output/dataset/scene), 'sfm_points': len(points),
                       'train_names': [c.name for c in train], 'test_names': [c.name for c in test],
                       'image_sizes': sorted({(c.width,c.height) for c in train+test}), 'status': 'queued'})
    snapshot = output/'source_snapshot'
    hashes = {}
    # Runtime modules and entry points are retained for reproducibility.
    files = [ROOT/name for name in ('train.py','compress.py','render.py')]
    for directory in ('arguments','scene','smolgs','utils','gaussian_renderer','scripts'):
        files.extend((ROOT/directory).rglob('*.py'))
    for source in files:
        relative = source.relative_to(ROOT)
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(config_path, output/'config.json')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'gpu': args.gpu,
                'pid': os.getpid(), 'python': sys.executable, 'config': config.to_dict(),
                'fresh': True, 'log_interval': 1, 'iterations_per_scene': 35000,
                'selection': 'opacity', 'source_sha256': hashes, 'scenes': scenes,
                'psnr_definition': 'per-step current training view; periodic test evaluation is separate'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    def status():
        (output/'status.json').write_text(json.dumps({'updated_utc': datetime.now(timezone.utc).isoformat(),
              'scenes': scenes}, indent=2)+'\n')

    def run(command, destination):
        Path(str(destination)+'.command.json').write_text(json.dumps(command, indent=2)+'\n')
        with destination.open('x') as log:
            subprocess.run(command, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT, check=True)

    status()
    for item in scenes:
        branch = Path(item['output'])
        branch.mkdir(parents=True, exist_ok=False)
        item['status'] = 'training'
        status()
        print(f"[Batch] training {item['dataset']}/{item['scene']}", flush=True)
        try:
            run([sys.executable, str(snapshot/'train.py'), '-s', item['source'], '-m', str(branch),
                 '--config', str(output/'config.json'), '-i', 'images', '-r', '1', '--holdout', '8',
                 '--seed', '0', '--iterations', '35000', '--log-interval', '1',
                 '--checkpoint-interval', '1000', '--coordinate-selection', 'opacity', '--codec', 'cuda',
                 '--eval-lpips', '--test-iterations', '7000','15000','20000','30000','35000'], branch/'training.log')
            item['status'] = 'validating'
            status()
            archive = branch/'model.smolgs.zip'
            run([sys.executable, str(snapshot/'scripts/audit_training_log.py'), '--model', str(branch)], branch/'training_log_audit.log')
            run([sys.executable, str(snapshot/'compress.py'), 'decode', str(archive), str(branch/'decoded.pt')], branch/'standalone_decode.log')
            run([sys.executable, str(snapshot/'scripts/verify_roundtrip.py'), '--checkpoint', str(branch/'checkpoint_35000.pt'),
                 '--archive', str(archive), '--output', str(branch/'roundtrip.json'), '--views', str(len(item['test_names']))], branch/'roundtrip.log')
            run([sys.executable, str(snapshot/'render.py'), '-m', str(archive), '-s', item['source'],
                 '-i','images','-r','1','--holdout','8','-o',str(branch/'decoded_evaluation')], branch/'decoded_evaluation.log')
            metrics = json.loads((branch/'decoded_evaluation/metrics.json').read_text())
            assert [r['name'] for r in metrics['per_view']] == item['test_names']
            item['result'] = {k: metrics[k] for k in ('psnr','ssim','lpips','views','splats')}
            item['result']['archive_bytes'] = archive.stat().st_size
            item['status'] = 'complete'
        except Exception:
            item['status'] = 'failed'
            item['error'] = traceback.format_exc()
            print(item['error'], flush=True)
        status()
    (output/'final_results.json').write_text(json.dumps(scenes, indent=2)+'\n')
    print('[Batch] finished', flush=True)
    if any(item['status'] != 'complete' for item in scenes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
