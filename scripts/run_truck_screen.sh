#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/Smol-GS
export TZ=Asia/Shanghai
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$PWD/.cuda124"
export PATH="$PWD/.env/bin:$CUDA_HOME/bin:$PATH"
SMOLGS_RUN_ROOT="$PWD/outputs/truck_sizeaware_base_20261021"
if [[ -e "$SMOLGS_RUN_ROOT" ]]; then
  echo "Output already exists: $SMOLGS_RUN_ROOT; choose a fresh run directory before starting." >&2
  exit 1
fi
mkdir -p "$SMOLGS_RUN_ROOT"
rm -f "$SMOLGS_RUN_ROOT/exit_code"
exec > >(tee -a "$SMOLGS_RUN_ROOT/run.log") 2>&1
trap 'smolgs_rc=$?; echo "[$(date -Is)] FAILED exit=$smolgs_rc line=$LINENO"; echo "$smolgs_rc" > "$SMOLGS_RUN_ROOT/exit_code"' ERR
printf '[%s] Truck only, base, 35000 iterations, screen-space XY + size-aware ADC, GPU 0, fresh run\n' "$(date -Is)"
.env/bin/python scripts/check_environment.py --output "$SMOLGS_RUN_ROOT/environment.json"
.env/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,shutil
root=Path.cwd()
run=root/'outputs/truck_sizeaware_base_20261021'
env=json.loads((run/'environment.json').read_text())
assert env['cuda_available'] and all(row['available'] for row in env['extensions'].values()), env
files=['train.py','scene/gaussian_model.py','gaussian_renderer/__init__.py','arguments/__init__.py',
       'utils/loss_utils.py','utils/checkpoint.py','smolgs/codec.py','scripts/run_benchmark.py',
       'scripts/verify_roundtrip.py','scripts/run_truck_screen.sh','configs/base.json',
       'docs/REFERENCE_ADC_AUDIT.md']
snapshot=run/'source'
manifest={}
for name in files:
    source=root/name
    target=snapshot/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
(run/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
.env/bin/python full_eval.py --dataset tandt --scene truck \
  --data-root /root/autodl-tmp/data --output-root "$SMOLGS_RUN_ROOT" \
  --variant base --gpu 0 --log-interval 50
.env/bin/python metrics.py -m "$SMOLGS_RUN_ROOT/tandt/truck/base"
.env/bin/python scripts/compare_tnt_db.py --output-root "$SMOLGS_RUN_ROOT"
printf '[%s] Truck completed: compression, roundtrip validation, decoded renders, float/PNG metrics, paper comparison\n' "$(date -Is)"
echo 0 > "$SMOLGS_RUN_ROOT/exit_code"
