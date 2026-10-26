#!/usr/bin/env bash
# Re-export a completed Truck checkpoint and finish verification/evaluation.
set -Eeuo pipefail
cd /root/autodl-tmp/Smol-GS
export TZ=Asia/Shanghai
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$PWD/.cuda124"
export PATH="$PWD/.env/bin:$CUDA_HOME/bin:$PATH"
SMOLGS_RUN_ROOT="$PWD/outputs/truck_sizeaware_base_20261021"
SMOLGS_MODEL_ROOT="$SMOLGS_RUN_ROOT/tandt/truck/base"
exec > >(tee -a "$SMOLGS_RUN_ROOT/run.log" "$SMOLGS_RUN_ROOT/postprocess.log") 2>&1
trap 'smolgs_rc=$?; echo "[$(date -Is)] POSTPROCESS FAILED exit=$smolgs_rc line=$LINENO"; echo "$smolgs_rc" > "$SMOLGS_RUN_ROOT/exit_code"' ERR
printf '[%s] Re-export completed Truck; full-scene entropy context, archive v3; no training\n' "$(date -Is)"
.env/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,shutil,time,zipfile
run=Path('outputs/truck_sizeaware_base_20261021')
model=run/'tandt/truck/base'
info=json.loads(Path(str(model/'checkpoint_latest.pt')+'.json').read_text())
assert info['step']==info['iterations']==35000, info
archive=model/'model.smolgs.zip'
with zipfile.ZipFile(archive) as source:
    version=json.loads(source.read('metadata.json'))['format']
if version in ('smolgs-archive-v1','smolgs-archive-v2'):
    backup=model/'model.before_context_fix.smolgs.zip'
    if not backup.exists():
        shutil.copy2(archive,backup)
if not (model/'summary.before_context_fix.json').exists():
    shutil.copy2(model/'summary.json',model/'summary.before_context_fix.json')
previous=run/'exit_code.before_context_fix'
if (run/'exit_code').exists():
    if not previous.exists():
        shutil.copy2(run/'exit_code',previous)
    (run/'exit_code').unlink()
snapshot=run/'postprocess_source'/time.strftime('%Y%m%d_%H%M%S')
files=['smolgs/codec.py','compress.py','scripts/verify_roundtrip.py','tests/test_core.py',
       'scripts/finish_truck_after_codec_fix.sh']
hashes={}
for name in files:
    target=snapshot/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(name,target)
    hashes[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
(snapshot/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('Preserved original archive, summary, and training source; postprocess source:',snapshot,flush=True)
PY
.env/bin/python compress.py encode "$SMOLGS_MODEL_ROOT/checkpoint_latest.pt" "$SMOLGS_MODEL_ROOT/model.smolgs.zip" --device cuda --codec cuda
.env/bin/python - <<'PY'
from pathlib import Path
import json
model=Path('outputs/truck_sizeaware_base_20261021/tandt/truck/base')
summary=json.loads((model/'summary.json').read_text())
summary['compression']=json.loads((model/'model.smolgs.zip.json').read_text())
summary['postprocess_note']='Re-exported the completed checkpoint with archive v3/full-scene entropy context; no training changes.'
(model/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
PY
.env/bin/python full_eval.py --dataset tandt --scene truck \
  --data-root /root/autodl-tmp/data --output-root "$SMOLGS_RUN_ROOT" \
  --variant base --gpu 0 --resume --log-interval 50
.env/bin/python metrics.py -m "$SMOLGS_MODEL_ROOT"
.env/bin/python scripts/compare_tnt_db.py --output-root "$SMOLGS_RUN_ROOT"
printf '[%s] POSTPROCESS SUCCESS: codec checks, decoded renders, float/PNG metrics, paper comparison\n' "$(date -Is)"
echo 0 > "$SMOLGS_RUN_ROOT/exit_code"
