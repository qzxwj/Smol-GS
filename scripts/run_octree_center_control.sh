#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/Smol-GS
export TZ=Asia/Shanghai PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
root="$PWD/outputs/truck_octree_fix_20261022/diagnosis"
exec > >(tee -a "$root/center_control.log") 2>&1
trap 'rc=$?; echo "$rc" > "$root/center_control_exit_code"' EXIT
printf '[%s] Queued for an idle GPU after the full default-weight control.\n' "$(date -Is)"
while [[ ! -f outputs/truck_default_nosize_20261022/exit_code ]]; do sleep 5; done
.env/bin/python - <<'SNAPSHOT'
from pathlib import Path
import hashlib,json,shutil
repo=Path.cwd();root=repo/'outputs/truck_octree_fix_20261022/diagnosis'
assert not (root/'torch_center05').exists(), 'Refusing to overwrite the queued centering experiment'
files=[p for directory in ('arguments','scene','gaussian_renderer','utils','smolgs','scripts') for p in (repo/directory).rglob('*.py')]
files+=[repo/'train.py',repo/'scripts/run_octree_center_control.sh']
manifest={}
for source in files:
 name=str(source.relative_to(repo));dest=root/'center_runtime_source'/name
 dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
 manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
(root/'center_runtime_source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
SNAPSHOT
.env/bin/python scripts/diagnose_octree_training.py --initialization torch --input-center 0.5 \
 -s /root/autodl-tmp/data/tandt/truck -m "$root/torch_center05" -r 1 --variant base \
 --seed 0 --stop-after 1000 --test-iterations 1000 --checkpoint-interval 1000 --skip-export --log-interval 50
.env/bin/python scripts/inspect_octree_checkpoint.py --checkpoint "$root/torch_center05/checkpoint_1000.pt" --output "$root/torch_center05/point_activity.json"
