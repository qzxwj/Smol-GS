#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/Smol-GS
export TZ=Asia/Shanghai PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_HOME="$PWD/.cuda124"
export PATH="$PWD/.env/bin:$CUDA_HOME/bin:$PATH"
SMOLGS_OCTREE_ROOT="$PWD/outputs/truck_octree_fix_20261022"
SMOLGS_OCTREE_MODEL="$SMOLGS_OCTREE_ROOT/tandt/truck/base"
if [[ -e "$SMOLGS_OCTREE_MODEL" || -e "$SMOLGS_OCTREE_ROOT/source_sha256.json" ]]; then
 echo 'This run already exists; refusing to overwrite it.' >&2
 exit 1
fi
mkdir -p "$SMOLGS_OCTREE_ROOT"
exec > >(tee -a "$SMOLGS_OCTREE_ROOT/run.log") 2>&1
trap 'smolgs_rc=$?; printf "[%s] exit=%s\n" "$(date -Is)" "$smolgs_rc"; echo "$smolgs_rc" > "$SMOLGS_OCTREE_ROOT/exit_code"' EXIT
printf '[%s] Truck base: fresh initialization, ReLU MLP, paper LRs and loss weights; size pruning ends at15k\n' "$(date -Is)"
.env/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,shutil
root=Path.cwd(); run=root/'outputs/truck_octree_fix_20261022'
files=[p for folder in ('arguments','scene','gaussian_renderer','utils','smolgs','scripts','tests')
       for p in (root/folder).rglob('*.py')]
files += [root/name for name in ('train.py','render.py','metrics.py','compress.py','configs/base.json',
                                 'scripts/run_truck_octree_fix_screen.sh','docs/FIDELITY.md')]
manifest={}
for source in files:
 name=str(source.relative_to(root));target=run/'source'/name
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
(run/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
(run/'experiment.json').write_text(json.dumps({
 'scene':'truck','seed':0,'lambda_opacity':3e-7,'lambda_rate':5e-4,
 'octree_init':'kaiming','octree_init_seed':1729,
 'architecture':'48->64 ReLU->64 ReLU->8; unchanged',
 'learning_rates':'paper Table A5; unchanged',
 'adc_size_prune_from':3000,'adc_size_prune_until':15000,
 'opacity_invisibility_finite_pruning':'unchanged',
 'gate':'all-point positional activity check at7000; per-log monitoring with full-point dead-MLP failure check',
 'scope':'fresh Truck only, 35000 steps, encode, roundtrip, decoded metrics',
 'initialization_caveat':'independent implementation choice; author MLP initialization is not disclosed'
},indent=2)+'\n')
PY
.env/bin/python scripts/check_environment.py --output "$SMOLGS_OCTREE_ROOT/environment.json"
.env/bin/python train.py -s /root/autodl-tmp/data/tandt/truck -m "$SMOLGS_OCTREE_MODEL" -r 1 \
 --variant base --seed 0 --octree-init kaiming --octree-init-seed 1729 --adc-size-prune-until 15000 \
 --lambda-opacity 3e-7 --lambda-rate 5e-4 --stop-after 7000 --test-iterations 7000 \
 --eval-lpips --log-interval 50 --monitor-octree --require-active-octree
.env/bin/python - <<'PY'
from pathlib import Path
import json,torch
from utils.checkpoint import load_checkpoint
root=Path('outputs/truck_octree_fix_20261022');p=root/'tandt/truck/base/checkpoint_7000.pt'
torch.set_num_threads(8)
model,_,step,_=load_checkpoint(p,'cuda')
health=model.octree_health(sample_size=None)
report={'step':step,**health,'lambda_opacity':model.config.lambda_opacity,
        'lambda_rate':model.config.lambda_rate,'adc_size_prune_until':model.config.adc_size_prune_until}
(root/'octree_gate_7000.json').write_text(json.dumps(report,indent=2)+'\n')
assert health['octree_second_active_fraction']>0 and health['octree_embedding_range']>0,report
assert model.config.lambda_opacity==3e-7 and model.config.lambda_rate==5e-4
print('[7k all-point gate passed] '+json.dumps(report),flush=True)
PY
.env/bin/python train.py -s /root/autodl-tmp/data/tandt/truck -m "$SMOLGS_OCTREE_MODEL" -r 1 \
 --resume "$SMOLGS_OCTREE_MODEL/checkpoint_7000.pt" --test-iterations 15000 20000 30000 35000 \
 --eval-lpips --log-interval 50 --monitor-octree --require-active-octree
.env/bin/python scripts/verify_roundtrip.py --checkpoint "$SMOLGS_OCTREE_MODEL/checkpoint_latest.pt" \
 --archive "$SMOLGS_OCTREE_MODEL/model.smolgs.zip" --output "$SMOLGS_OCTREE_MODEL/roundtrip.json"
.env/bin/python render.py -m "$SMOLGS_OCTREE_MODEL/model.smolgs.zip" -s /root/autodl-tmp/data/tandt/truck \
 -r 1 -o "$SMOLGS_OCTREE_MODEL/test/ours_35000" --save-images
.env/bin/python metrics.py -m "$SMOLGS_OCTREE_MODEL"
.env/bin/python scripts/collect_results.py --output-root "$SMOLGS_OCTREE_ROOT"
.env/bin/python scripts/compare_tnt_db.py --output-root "$SMOLGS_OCTREE_ROOT"
printf '[%s] COMPLETE: 35k, archive, roundtrip, decoded views and metrics\n' "$(date -Is)"
