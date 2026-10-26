#!/usr/bin/env bash
# Complete the size-rule diagnosis after the controller/effective comparison.
set -Eeuo pipefail
cd /root/autodl-tmp/Smol-GS
export TZ=Asia/Shanghai
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$PWD/.cuda124"
export PATH="$PWD/.env/bin:$CUDA_HOME/bin:$PATH"
SMOLGS_AUDIT_ROOT="$PWD/outputs/truck_compaction_audit_20261022"
SMOLGS_PARENT="$PWD/outputs/truck_sizeaware_base_20261021/tandt/truck/base/checkpoint_15000.pt"
exec > >(tee -a "$SMOLGS_AUDIT_ROOT/run.log" "$SMOLGS_AUDIT_ROOT/no_size_followup.log") 2>&1
printf '[%s] Queued compaction-only size-rule omission; waiting for the two-arm diagnosis\n' "$(date -Is)"
while [[ ! -f "$SMOLGS_AUDIT_ROOT/exit_code" ]]; do
  sleep 5
done
if [[ "$(cat "$SMOLGS_AUDIT_ROOT/exit_code")" != 0 ]]; then
  echo 'Earlier audit failed; follow-up not started.'
  exit 1
fi
mv "$SMOLGS_AUDIT_ROOT/exit_code" "$SMOLGS_AUDIT_ROOT/exit_code.two_arm"
trap 'smolgs_rc=$?; echo "[$(date -Is)] FOLLOWUP FAILED exit=$smolgs_rc line=$LINENO"; echo "$smolgs_rc" > "$SMOLGS_AUDIT_ROOT/exit_code"' ERR
.env/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,shutil
root=Path('outputs/truck_compaction_audit_20261022')
for name in ['report.json','report.md']:
    if (root/name).exists():shutil.copy2(root/name,root/(name+'.two_arm'))
files=['scripts/audit_compaction.py','scripts/summarize_compaction_audit.py',
       'scripts/run_compaction_no_size_followup.sh','scene/gaussian_model.py']
manifest={}
for name in files:
    target=root/'followup_source'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(name,target)
    manifest[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
(root/'followup_source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'followup_decision.json').write_text(json.dumps({
    'reason':'Same-weight measurements show damage from size-only deletion; all-view false-positive corrections explain only a small fraction.',
    'change':'Omit the extra size rule only during15k--20k; keep opacity/invisibility/nonfinite pruning, parent/Adam/RNG, weights, learning rates and every earlier stage unchanged.',
    'lambda_opacity':3e-7,'lambda_rate':5e-4,'stop':20000,
    'note':'No35k quality or archive-size claim follows from this20k diagnosis.'},indent=2)+'\n')
PY
printf '[%s] Starting no-extra-size compaction control, same15k checkpoint/default weights\n' "$(date -Is)"
.env/bin/python scripts/audit_compaction.py --checkpoint "$SMOLGS_PARENT" --output "$SMOLGS_AUDIT_ROOT/no_size" --mode no_size
.env/bin/python scripts/summarize_compaction_audit.py --root "$SMOLGS_AUDIT_ROOT"
printf '[%s] THREE-ARM AUDIT SUCCESS; all runs stopped at20k\n' "$(date -Is)"
echo 0 > "$SMOLGS_AUDIT_ROOT/exit_code"
