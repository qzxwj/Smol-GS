#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/Smol-GS
export TZ=Asia/Shanghai PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_HOME="$PWD/.cuda124"
export PATH="$PWD/.env/bin:$CUDA_HOME/bin:$PATH"
SMOLGS_DUAL_PILOT="$PWD/outputs/truck_relu_dual_pilot_20261022"
if [[ -e "$SMOLGS_DUAL_PILOT" ]]; then echo 'Fallback directory already exists; refusing overwrite.' >&2; exit 1; fi
mkdir -p "$SMOLGS_DUAL_PILOT"
exec > >(tee -a "$SMOLGS_DUAL_PILOT/run.log") 2>&1
trap 'smolgs_rc=$?; echo "$smolgs_rc" > "$SMOLGS_DUAL_PILOT/exit_code"; printf "[%s] exit=%s\n" "$(date -Is)" "$smolgs_rc"' EXIT
printf '[%s] Waiting for single-input-center gates; activate fallback only on a diagnosed positional failure.\n' "$(date -Is)"
while [[ ! -f outputs/truck_relu_center05_20261022/exit_code ]]; do sleep 5; done
if .env/bin/python - <<'PY'
from pathlib import Path
import json
root=Path('outputs/truck_relu_center05_20261022')
code=int((root/'exit_code').read_text().strip())
failed=[str(path) for path in root.glob('gate_*.json') if not json.loads(path.read_text())['passed']]
dead=(root/'tandt/truck/base/octree_failure.json').exists()
r={'parent_exit_code':code,'failed_activity_gates':failed,'confirmed_dead_positional_mlp':dead,
   'activate_fallback':code!=0 and (bool(failed) or dead)}
Path('outputs/truck_relu_dual_pilot_20261022/decision.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r),flush=True)
raise SystemExit(0 if r['activate_fallback'] else 3)
PY
then
 printf '[%s] Diagnosed positional failure; starting the second affine parameterization control.\n' "$(date -Is)"
else
 printf '[%s] No diagnosed positional failure; fallback skipped.\n' "$(date -Is)"
 exit 0
fi
.env/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,shutil
repo=Path.cwd();out=repo/'outputs/truck_relu_dual_pilot_20261022'
files=[f for directory in ('arguments','scene','gaussian_renderer','utils','smolgs','scripts','tests') for f in (repo/directory).rglob('*.py')]
files += [repo/'train.py',repo/'scripts/run_truck_relu_compatible_fallback.sh',repo/'scripts/run_truck_relu_dual_center_screen.sh']
manifest={}
for source in files:
 name=str(source.relative_to(repo));dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
 manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
(out/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
.env/bin/python -m unittest discover -s tests -v > "$SMOLGS_DUAL_PILOT/tests_gpu.log" 2>&1
.env/bin/python train.py -s /root/autodl-tmp/data/tandt/truck -m "$SMOLGS_DUAL_PILOT/model" -r 1 --variant base --seed 0 \
 --octree-input-center 0.5 --octree-hidden-center-mode sfm --adc-size-prune-until 15000 \
 --lambda-opacity 3e-7 --lambda-rate 5e-4 --stop-after 1000 --test-iterations 1000 --eval-lpips \
 --log-interval 50 --monitor-octree --require-active-octree
.env/bin/python scripts/inspect_octree_checkpoint.py --checkpoint "$SMOLGS_DUAL_PILOT/model/checkpoint_1000.pt" \
 --output "$SMOLGS_DUAL_PILOT/activity_1000.json"
# The continuation still applies full-point gates at7k/15k/20k/30k/35k.
bash scripts/run_truck_relu_dual_center_screen.sh
