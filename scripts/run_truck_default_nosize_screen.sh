#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/Smol-GS
export TZ=Asia/Shanghai PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_HOME="$PWD/.cuda124"
export PATH="$PWD/.env/bin:$CUDA_HOME/bin:$PATH"
SMOLGS_CONTROL_ROOT="$PWD/outputs/truck_default_nosize_20261022"
SMOLGS_CONTROL_MODEL="$SMOLGS_CONTROL_ROOT/tandt/truck/base"
if [[ -e "$SMOLGS_CONTROL_ROOT" ]]; then echo 'Control already exists; refusing overwrite.' >&2; exit 1; fi
mkdir -p "$SMOLGS_CONTROL_ROOT"
exec > >(tee -a "$SMOLGS_CONTROL_ROOT/run.log") 2>&1
trap 'smolgs_rc=$?; echo "$smolgs_rc" > "$SMOLGS_CONTROL_ROOT/exit_code"; printf "[%s] exit=%s\n" "$(date -Is)" "$smolgs_rc"' EXIT
printf '[%s] Queued: finish validated no-size compaction branch, original ReLU, default lambdas\n' "$(date -Is)"
while [[ ! -f outputs/truck_octree_fix_20261022/diagnosis/eps_control_exit_code ]]; do sleep 5; done
.env/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,torch,shutil
from utils.checkpoint import load_torch
root=Path.cwd();run=root/'outputs/truck_default_nosize_20261022'
parent=root/'outputs/truck_compaction_audit_20261022/no_size/replay/checkpoint_20000.pt'
payload=load_torch(parent)
assert payload['step']==20000 and payload['config']['lambda_opacity']==3e-7 and payload['config']['lambda_rate']==5e-4
assert payload['config'].get('octree_activation','relu')=='relu'
# Parent audit disabled only the compaction size criterion via a wrapper. Record
# that measured history explicitly; no ADC is scheduled after resumed step20k.
payload['config']['adc_size_prune_until']=15000
payload['metadata']['continuation_provenance']={'parent':str(parent),'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
 'config_annotation':'size-prune end reflects the parent audit policy; no model/optimizer/RNG tensors changed',
 'position_network':'original ReLU, known degeneration retained as an explicit control'}
torch.save(payload,run/'resume_20000.pt')
files=[p for directory in ('arguments','scene','gaussian_renderer','utils','smolgs','scripts','tests') for p in (root/directory).rglob('*.py')]
files += [root/p for p in ('train.py','render.py','metrics.py','scripts/run_truck_default_nosize_screen.sh')]
manifest={}
for source in files:
 name=str(source.relative_to(root));dest=run/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
 manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
(run/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
(run/'experiment.json').write_text(json.dumps({'purpose':'Isolate the final benefit of the validated15k--20k size-prune correction; not a repaired positional-MLP experiment',
 'parent_checkpoint':str(parent),'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
 'lambda_opacity':3e-7,'lambda_rate':5e-4,'octree_activation':'relu',
 'position_network_caveat':'Known dead positional MLP intentionally retained; no initialization or activation repair applied',
 'start':20000,'stop':35000},indent=2)+'\n')
PY
.env/bin/python train.py -s /root/autodl-tmp/data/tandt/truck -m "$SMOLGS_CONTROL_MODEL" -r 1 \
 --resume "$SMOLGS_CONTROL_ROOT/resume_20000.pt" --test-iterations 30000 35000 --eval-lpips \
 --log-interval 50 --monitor-octree
.env/bin/python scripts/verify_roundtrip.py --checkpoint "$SMOLGS_CONTROL_MODEL/checkpoint_latest.pt" \
 --archive "$SMOLGS_CONTROL_MODEL/model.smolgs.zip" --output "$SMOLGS_CONTROL_MODEL/roundtrip.json"
.env/bin/python render.py -m "$SMOLGS_CONTROL_MODEL/model.smolgs.zip" -s /root/autodl-tmp/data/tandt/truck \
 -r 1 -o "$SMOLGS_CONTROL_MODEL/test/ours_35000" --save-images
.env/bin/python metrics.py -m "$SMOLGS_CONTROL_MODEL"
.env/bin/python scripts/collect_results.py --output-root "$SMOLGS_CONTROL_ROOT"
.env/bin/python scripts/compare_tnt_db.py --output-root "$SMOLGS_CONTROL_ROOT"
printf '[%s] COMPLETE default-weight ReLU control:35k, archive, roundtrip and decoded metrics\n' "$(date -Is)"
