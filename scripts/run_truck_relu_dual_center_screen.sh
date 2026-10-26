#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/Smol-GS
export TZ=Asia/Shanghai PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_HOME="$PWD/.cuda124"
export PATH="$PWD/.env/bin:$CUDA_HOME/bin:$PATH"
SMOLGS_RELU_ROOT="$PWD/outputs/truck_relu_dual_center_20261022"
SMOLGS_RELU_MODEL="$SMOLGS_RELU_ROOT/tandt/truck/base"
if [[ -e "$SMOLGS_RELU_ROOT" ]]; then echo 'Output exists; refusing overwrite.' >&2; exit 1; fi
mkdir -p "$SMOLGS_RELU_ROOT"
exec > >(tee -a "$SMOLGS_RELU_ROOT/run.log") 2>&1
trap 'smolgs_rc=$?; echo "$smolgs_rc" > "$SMOLGS_RELU_ROOT/exit_code"; printf "[%s] exit=%s\n" "$(date -Is)" "$smolgs_rc"' EXIT
printf '[%s] ReLU-compatible input-centering continuation; fixed paper rates and loss weights\n' "$(date -Is)"
.env/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,shutil,torch
from utils.checkpoint import load_torch
root=Path.cwd();run=root/'outputs/truck_relu_dual_center_20261022'
parent=root/'outputs/truck_relu_dual_pilot_20261022/model/checkpoint_1000.pt'
p=load_torch(parent)
assert p['step']==1000 and p['config']['octree_activation']=='relu' and p['config']['octree_input_center']==0.5
assert p['config']['lambda_opacity']==3e-7 and p['config']['lambda_rate']==5e-4
assert p['config']['octree_hidden_center_mode']=='sfm' and len(p['config']['octree_hidden_center'])==64
p['config']['adc_size_prune_until']=15000
p['metadata']['continuation_provenance']={'parent':str(parent),'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
 'edit':'Only future size-pruning end set to15000; model, optimizer and RNG unchanged'}
torch.save(p,run/'resume_1000.pt')
files=[f for folder in ('arguments','scene','gaussian_renderer','utils','smolgs','scripts','tests') for f in (root/folder).rglob('*.py')]
files += [root/'train.py',root/'render.py',root/'metrics.py',root/'scripts/run_truck_relu_dual_center_screen.sh']
manifest={}
for source in files:
 name=str(source.relative_to(root));target=run/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
(run/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
(run/'experiment.json').write_text(json.dumps({'route':'User explicitly selected paper ReLU; no activation substitution',
 'lambda_opacity':3e-7,'lambda_rate':5e-4,'learning_rates':'unchanged Table A5','octree_input_center':0.5,
 'octree_hidden_center_mode':p['config']['octree_hidden_center_mode'],
 'octree_hidden_center':p['config']['octree_hidden_center'],
 'architecture':'unchanged affine/ReLU function family; both first and second bias compensations preserve initial function',
 'caveat':'optimizer parameterization differs; author implementation not confirmed',
 'parent_checkpoint':str(parent),'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
 'point_activity_gate':{'all_min_fraction':0.10,'near_camera_mean_min_fraction':0.10,'max_channel_std_min':0.005},
 'gate_note':'Engineering diagnostic stopping criteria, not paper hyperparameters or a training loss; no quality metric is used for gating'},indent=2)+'\n')
PY
SMOLGS_RELU_PARENT="$SMOLGS_RELU_ROOT/resume_1000.pt"
for smolgs_target in 7000 15000 20000 30000 35000; do
 .env/bin/python train.py -s /root/autodl-tmp/data/tandt/truck -m "$SMOLGS_RELU_MODEL" -r 1 \
  --resume "$SMOLGS_RELU_PARENT" --stop-after "$smolgs_target" --test-iterations "$smolgs_target" \
  --eval-lpips --log-interval 50 --monitor-octree --require-active-octree
 .env/bin/python scripts/inspect_octree_checkpoint.py --checkpoint "$SMOLGS_RELU_MODEL/checkpoint_${smolgs_target}.pt" \
  --output "$SMOLGS_RELU_ROOT/activity_${smolgs_target}.json"
 if .env/bin/python - "$smolgs_target" <<'PY'
from pathlib import Path
import json,sys
root=Path('outputs/truck_relu_dual_center_20261022');step=int(sys.argv[1])
r=json.loads((root/f'activity_{step}.json').read_text());a=r['groups']['all'];n=r['groups']['near_camera_mean']
passed=a['active_point_fraction']>=.10 and n['active_point_fraction']>=.10 and max(a['embedding_std_per_channel'])>=.005
report={'step':step,'passed':passed,'all_active_fraction':a['active_point_fraction'],'near_active_fraction':n['active_point_fraction'],
        'max_embedding_std':max(a['embedding_std_per_channel']),'decision':'continue' if passed else 'reject candidate; keep checkpoint'}
(root/f'gate_{step}.json').write_text(json.dumps(report,indent=2)+'\n');print('[Point-activity gate] '+json.dumps(report),flush=True)
raise SystemExit(0 if passed else 2)
PY
 then
  if [[ "$smolgs_target" == 7000 || "$smolgs_target" == 35000 ]]; then
   .env/bin/python scripts/audit_octree_embedding_effect.py --checkpoint "$SMOLGS_RELU_MODEL/checkpoint_${smolgs_target}.pt" \
    --output "$SMOLGS_RELU_ROOT/embedding_effect_${smolgs_target}"
  fi
 else
  printf '[%s] Candidate rejected at%s; no automatic longer run.\n' "$(date -Is)" "$smolgs_target"
  exit 2
 fi
 SMOLGS_RELU_PARENT="$SMOLGS_RELU_MODEL/checkpoint_${smolgs_target}.pt"
done
.env/bin/python scripts/verify_roundtrip.py --checkpoint "$SMOLGS_RELU_PARENT" \
 --archive "$SMOLGS_RELU_MODEL/model.smolgs.zip" --output "$SMOLGS_RELU_MODEL/roundtrip.json"
.env/bin/python render.py -m "$SMOLGS_RELU_MODEL/model.smolgs.zip" -s /root/autodl-tmp/data/tandt/truck \
 -r 1 -o "$SMOLGS_RELU_MODEL/test/ours_35000" --save-images
.env/bin/python metrics.py -m "$SMOLGS_RELU_MODEL"
.env/bin/python scripts/collect_results.py --output-root "$SMOLGS_RELU_ROOT"
.env/bin/python scripts/compare_tnt_db.py --output-root "$SMOLGS_RELU_ROOT"
printf '[%s] COMPLETE ReLU input-centering experiment\n' "$(date -Is)"
