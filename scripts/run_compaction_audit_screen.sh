#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/Smol-GS
export TZ=Asia/Shanghai
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$PWD/.cuda124"
export PATH="$PWD/.env/bin:$CUDA_HOME/bin:$PATH"
SMOLGS_AUDIT_ROOT="$PWD/outputs/truck_compaction_audit_20261022"
SMOLGS_PARENT="$PWD/outputs/truck_sizeaware_base_20261021/tandt/truck/base/checkpoint_15000.pt"
if [[ -e "$SMOLGS_AUDIT_ROOT" ]]; then
  echo "Refusing to overwrite $SMOLGS_AUDIT_ROOT" >&2
  exit 1
fi
mkdir -p "$SMOLGS_AUDIT_ROOT"
exec > >(tee -a "$SMOLGS_AUDIT_ROOT/run.log") 2>&1
trap 'smolgs_rc=$?; echo "[$(date -Is)] AUDIT FAILED exit=$smolgs_rc line=$LINENO"; echo "$smolgs_rc" > "$SMOLGS_AUDIT_ROOT/exit_code"' ERR
printf '[%s] Default-weight Truck compaction audit, 15k--20k only\n' "$(date -Is)"
.env/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,shutil
from utils.checkpoint import load_torch
root=Path('outputs/truck_compaction_audit_20261022')
parent=Path('outputs/truck_sizeaware_base_20261021/tandt/truck/base/checkpoint_15000.pt')
p=load_torch(parent)
assert p['step']==15000 and p['config']['lambda_opacity']==3e-7 and p['config']['lambda_rate']==5e-4
files=['train.py','scene/gaussian_model.py','gaussian_renderer/__init__.py','arguments/__init__.py',
       'utils/loss_utils.py','utils/checkpoint.py','utils/graphics_utils.py','scene/dataset_readers.py',
       'scripts/audit_compaction.py','scripts/run_compaction_audit_screen.sh','tests/test_core.py']
manifest={}
for name in files:
    target=root/'source'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(name,target)
    manifest[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
(root/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'README.md').write_text('''# Truck compaction audit, default paper weights

lambda_opacity=3e-7, lambda_rate=5e-4. Same original 15k model/Adam/RNG state; stop at20k.
Baseline retains the original controller-size rule. Every ADC logs mutually exclusive reason
combinations, effective axes across all219 training cameras and four fixed test-view probes.
Deletion and refreshed octree bounds are compared separately at identical weights.
Full32-view PSNR/SSIM/VGG-LPIPS counterfactuals cover the largest probe drop and final ADC.
If size-only points below the physical limit are found, a second replay changes only the
size criterion to maximum effective axes across all training cameras. Neither run continues to35k.
Per-point original15k IDs and candidate measurements are stored in compressed NPZ files.
No original experiment is rewritten. Diagnostic RNG preservation is asserted at every ADC.

screen -x smolgs_truck_prune_audit
tail -n20 -F /root/autodl-tmp/Smol-GS/outputs/truck_compaction_audit_20261022/run.log

exit_code=0 means the full investigation finished. Read report.md and report.json.
''')
print('Original15k checkpoint and default coefficients verified.',flush=True)
PY
.env/bin/python scripts/audit_compaction.py --checkpoint "$SMOLGS_PARENT" --output "$SMOLGS_AUDIT_ROOT/baseline" --mode baseline
SMOLGS_NEEDS_EFFECTIVE=$(.env/bin/python - <<'PY'
from pathlib import Path
import json
p=Path('outputs/truck_compaction_audit_20261022/baseline/audit_summary.json')
print(int(json.loads(p.read_text())['size_only_physically_small_total']>0))
PY
)
if [[ "$SMOLGS_NEEDS_EFFECTIVE" == 1 ]]; then
  printf '[%s] Size-only physically-small candidates found; replaying only the corrected size criterion\n' "$(date -Is)"
  .env/bin/python scripts/audit_compaction.py --checkpoint "$SMOLGS_PARENT" --output "$SMOLGS_AUDIT_ROOT/effective" --mode effective
fi
.env/bin/python scripts/summarize_compaction_audit.py --root "$SMOLGS_AUDIT_ROOT"
printf '[%s] AUDIT SUCCESS; stopped at20k, no compression-stage training\n' "$(date -Is)"
echo 0 > "$SMOLGS_AUDIT_ROOT/exit_code"
