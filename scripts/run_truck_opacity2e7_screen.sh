#!/usr/bin/env bash
# Opacity ablation against the rate2e4 run: restart 15k--35k, keep rate2e4.
set -Eeuo pipefail
cd /root/autodl-tmp/Smol-GS
export TZ=Asia/Shanghai
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$PWD/.cuda124"
export PATH="$PWD/.env/bin:$CUDA_HOME/bin:$PATH"
SMOLGS_RUN_ROOT="$PWD/outputs/truck_opacity2e7_rate2e4_20261022"
SMOLGS_MODEL_ROOT="$SMOLGS_RUN_ROOT/tandt/truck/base"
if [[ -e "$SMOLGS_RUN_ROOT" ]]; then
  echo "Output already exists: $SMOLGS_RUN_ROOT; refusing to overwrite." >&2
  exit 1
fi
mkdir -p "$SMOLGS_RUN_ROOT"
exec > >(tee -a "$SMOLGS_RUN_ROOT/run.log") 2>&1
trap 'smolgs_rc=$?; echo "[$(date -Is)] FAILED exit=$smolgs_rc line=$LINENO"; echo "$smolgs_rc" > "$SMOLGS_RUN_ROOT/exit_code"' ERR
printf '[%s] Truck opacity ablation: lambda_opacity 3e-7 -> 2e-7, lambda_rate 2e-4, same 15k state, finish at 35k\n' "$(date -Is)"
.env/bin/python scripts/check_environment.py --output "$SMOLGS_RUN_ROOT/environment.json"
.env/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import torch
from arguments import Config
from utils.checkpoint import load_torch

root=Path.cwd()
run=root/'outputs/truck_opacity2e7_rate2e4_20261022'
parent=root/'outputs/truck_sizeaware_base_20261021/tandt/truck/base/checkpoint_15000.pt'
payload=load_torch(parent)
assert payload['step']==payload['config']['densify_until']==15000
assert payload['config']['lambda_rate']==5e-4
assert payload['config']['lambda_opacity']==3e-7
assert not bool(payload['model']['coordinate_frozen'])
before=dict(payload['config'])
payload['config']={**before,'lambda_rate':2e-4,'lambda_opacity':2e-7}
Config(**payload['config'])
experiment={'type':'opacity-ablation-at-fixed-rate2e4','parent_checkpoint':str(parent),
            'comparison_baseline':str(root/'outputs/truck_rate2e4_20261022'),
            'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
            'resume_step':15000,'finish_step':35000,'target_size_MB_reference':5.369026,
            'overrides':{'lambda_rate':{'before':5e-4,'after':2e-4},'lambda_opacity':{'before':3e-7,'after':2e-7}},
            'note':'Same model, optimizer and RNG states at 15k. Compared with the completed rate2e4 trial, only opacity regularization changes; rate is inactive before 20k. Densification, pruning rules, quantization steps, learning rates and stage durations are unchanged. Size is measured, not enforced.'}
payload['metadata']={**payload['metadata'],'opacity_ablation':experiment}
fork=run/'checkpoint_15000_opacity2e7_rate2e4.pt'
torch.save(payload,fork)
check=load_torch(fork)
original=load_torch(parent)
def same(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
for key in ('model','optimizer','rng','step','format'):
    assert same(original[key],check[key]),key
assert {key for key in original['config'] if original['config'][key]!=check['config'][key]}=={'lambda_rate','lambda_opacity'}
experiment['fork_state_verified']=True
Path(str(fork)+'.json').write_text(json.dumps({'step':15000,'iterations':35000,
    'splats':len(check['model']['xyz']),'format':check['format']},indent=2)+'\n')
(run/'experiment.json').write_text(json.dumps(experiment,indent=2)+'\n')
(run/'trial_config.json').write_text(json.dumps(check['config'],indent=2)+'\n')
files=['train.py','scene/gaussian_model.py','gaussian_renderer/__init__.py','arguments/__init__.py',
       'utils/loss_utils.py','utils/checkpoint.py','smolgs/codec.py','scripts/run_benchmark.py',
       'scripts/verify_roundtrip.py','scripts/run_truck_opacity2e7_screen.sh']
manifest={}
for name in files:
    target=run/'source'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/name,target)
    manifest[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
(run/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(experiment,indent=2),flush=True)
PY
.env/bin/python train.py -s /root/autodl-tmp/data/tandt/truck -m "$SMOLGS_MODEL_ROOT" \
  -i images -r 1 --eval --variant base --iterations 35000 --codec cuda --gpu 0 \
  --resume "$SMOLGS_RUN_ROOT/checkpoint_15000_opacity2e7_rate2e4.pt" --log-interval 50 --eval-lpips
.env/bin/python full_eval.py --dataset tandt --scene truck \
  --data-root /root/autodl-tmp/data --output-root "$SMOLGS_RUN_ROOT" \
  --variant base --gpu 0 --resume --log-interval 50
.env/bin/python metrics.py -m "$SMOLGS_MODEL_ROOT"
.env/bin/python scripts/compare_tnt_db.py --output-root "$SMOLGS_RUN_ROOT"
.env/bin/python - <<'PY'
from pathlib import Path
import json
root=Path('outputs/truck_opacity2e7_rate2e4_20261022')
baseline=Path('outputs/truck_rate2e4_20261022')
base=json.loads((baseline/'benchmark_results.json').read_text())['per_scene'][0]
trial=json.loads((root/'benchmark_results.json').read_text())['per_scene'][0]
comparison={'lambda_rate':2e-4,'baseline_lambda_opacity':3e-7,'trial_lambda_opacity':2e-7,'baseline':base,'trial':trial,
            'delta':{key:trial[key]-base[key] for key in ('psnr','ssim','lpips','splats','size_MB')}}
(root/'opacity_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
lines=['# Truck: opacity ablation at fixed lambda_rate=2e-4','',
       'Common 15k checkpoint and 35k endpoint; lambda_rate=2e-4 in both runs, only opacity regularization changed.',
       '', '| Metric | Baseline opacity 3e-7 | Trial opacity 2e-7 | Delta |','|---|---:|---:|---:|']
for key in ('psnr','ssim','lpips','splats','size_MB'):
    lines.append(f'| {key} | {base[key]:.6f} | {trial[key]:.6f} | {comparison["delta"][key]:.6f} |')
(root/'opacity_comparison.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines),flush=True)
PY
printf '[%s] SUCCESS: Truck opacity ablation, roundtrip verification, decoded evaluation and comparisons\n' "$(date -Is)"
echo 0 > "$SMOLGS_RUN_ROOT/exit_code"
