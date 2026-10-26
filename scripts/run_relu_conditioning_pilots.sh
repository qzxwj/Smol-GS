#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/Smol-GS
export TZ=Asia/Shanghai PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
root="$PWD/outputs/truck_relu_conditioning_pilots_20261022"
if [[ -e "$root" ]]; then echo 'Pilot root exists; refusing overwrite.' >&2; exit 1; fi
mkdir -p "$root"
exec > >(tee -a "$root/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$root/exit_code"; printf "[%s] exit=%s\n" "$(date -Is)" "$rc"' EXIT
.env/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,shutil
repo=Path.cwd();out=repo/'outputs/truck_relu_conditioning_pilots_20261022'
files=[p for directory in ('arguments','scene','gaussian_renderer','utils','smolgs','scripts','tests') for p in (repo/directory).rglob('*.py')]
files += [repo/'train.py',repo/'scripts/run_relu_conditioning_pilots.sh']
manifest={}
for source in files:
 name=str(source.relative_to(repo));dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
 manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
(out/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
.env/bin/python train.py -s /root/autodl-tmp/data/tandt/truck -m "$root/sfm_means" -r 1 --variant base --seed 0 \
 --octree-input-center-mode sfm --octree-hidden-center-mode sfm --adc-size-prune-until 15000 \
 --lambda-opacity 3e-7 --lambda-rate 5e-4 --stop-after 1000 --test-iterations 1000 --eval-lpips --log-interval 50 --monitor-octree --require-active-octree
.env/bin/python scripts/inspect_octree_checkpoint.py --checkpoint "$root/sfm_means/checkpoint_1000.pt" --output "$root/sfm_means/activity_1000.json"
.env/bin/python train.py -s /root/autodl-tmp/data/tandt/truck -m "$root/gain8" -r 1 --variant base --seed 0 \
 --octree-init-gain 8 --adc-size-prune-until 15000 \
 --lambda-opacity 3e-7 --lambda-rate 5e-4 --stop-after 1000 --test-iterations 1000 --eval-lpips --log-interval 50 --monitor-octree --require-active-octree
.env/bin/python scripts/inspect_octree_checkpoint.py --checkpoint "$root/gain8/checkpoint_1000.pt" --output "$root/gain8/activity_1000.json"
