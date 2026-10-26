#!/usr/bin/env python3
"""Serial, isolated experiment for a function-preserving repair at iteration 20k."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--wait-for',required=True)
    p.add_argument('--conditioning',choices=['none','current_points'],default='none')
    args=p.parse_args()
    project=Path(args.project).resolve();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    def record(name,value):
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    record('status.json',{'status':'waiting_for_serial_gpu','waiting_for':args.wait_for})
    # Freeze execution as well as provenance: subsequent workspace edits cannot
    # change an already queued experiment's evaluation or training implementation.
    source=out/'source';manifest={}
    files=[f for folder in ('arguments','scene','gaussian_renderer','utils','smolgs','scripts','tests')
           for f in (project/folder).rglob('*.py')]
    files += [project/name for name in ('train.py','render.py','metrics.py')]
    for f in files:
        rel=f.relative_to(project);dest=source/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,dest);manifest[str(rel)]=hashlib.sha256(dest.read_bytes()).hexdigest()
    shutil.copytree(project/'official_snapshot/results',source/'official_snapshot/results')
    record('source_sha256.json',manifest)
    record('experiment.json',{
        'activation':'ReLU','lambda_opacity':3e-7,'lambda_rate':5e-4,
        'learning_rates':'Unchanged paper Table A5, global 30000-step schedule',
        'parent':'Original no-size-prune compaction checkpoint at 20000',
        'intervention':'Reset two dead hidden affine layers and zero the output weight; preserve the output bias',
        'optimizer':'Reset only five reinitialized parameter states',
        'scope':'Diagnostic engineering repair, not an author-confirmed paper training step',
        'conditioning':args.conditioning,
        'centering_note':'When selected, fixed input/hidden means come from the current 20k cloud, not the initial SfM cloud; affine/ReLU function family and learned parameter count are unchanged',
        'control':'Matched original 20k checkpoint continued to 21k with identical RNG and source',
        'gate':'Engineering stopping criteria, not a paper hyperparameter: >=10% active all/near points; max embedding std>=0.005',
        'execution':'All child processes execute the frozen source snapshot; one GPU process at a time'})
    waited=Path(args.wait_for)
    while not waited.exists():time.sleep(10)
    prior=waited.read_text().strip()
    if prior!='2':
        record('status.json',{'status':'skipped','reason':'Previous candidate did not exit with the documented activity-gate rejection','previous_exit':prior})
        return 0
    python=str(project/'.env/bin/python')
    env=dict(os.environ,PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8')
    def run(*cmd):
        print(f'[{datetime.now().isoformat()}] '+ ' '.join(map(str,cmd)),flush=True)
        subprocess.run([python,*map(str,cmd)],cwd=source,env=env,check=True)
    parent=project/'outputs/truck_compaction_audit_20261022/no_size/replay/checkpoint_20000.pt'
    repaired=out/'repaired_20000.pt';model=out/'tandt/truck/base';control=out/'matched_control'
    def train(checkpoint,folder,target,monitor=True):
        cmd=['train.py','-s','/root/autodl-tmp/data/tandt/truck','-m',folder,'-r','1',
             '--resume',checkpoint,'--stop-after',target,'--test-iterations',target,
             '--eval-lpips','--log-interval','50','--monitor-octree']
        if monitor:cmd+=['--require-active-octree']
        run(*cmd)
    record('status.json',{'status':'verifying_and_training'})
    run('-m','unittest','discover','-s','tests','-v')
    run('scripts/repair_dead_octree.py','--checkpoint',parent,'--output',repaired,'--conditioning',args.conditioning)
    train(parent,control,21000,False)
    current=repaired
    for target in (21000,25000,30000,35000):
        train(current,model,target)
        current=model/f'checkpoint_{target}.pt'
        activity=out/f'activity_{target}.json'
        run('scripts/inspect_octree_checkpoint.py','--checkpoint',current,'--output',activity)
        a=json.loads(activity.read_text())['groups'];g=a['all'];n=a['near_camera_mean']
        passed=g['active_point_fraction']>=.10 and n['active_point_fraction']>=.10 and max(g['embedding_std_per_channel'])>=.005
        gate={'step':target,'passed':passed,'all_active_fraction':g['active_point_fraction'],
              'near_active_fraction':n['active_point_fraction'],'max_embedding_std':max(g['embedding_std_per_channel'])}
        record(f'gate_{target}.json',gate);print('[Activity gate] '+json.dumps(gate),flush=True)
        if target in (21000,35000):
            run('scripts/audit_octree_embedding_effect.py','--checkpoint',current,'--output',out/f'embedding_effect_{target}')
        if target==21000:
            c=json.loads((control/'evaluation/iteration_21000/metrics.json').read_text())
            r=json.loads((model/'evaluation/iteration_21000/metrics.json').read_text())
            record('matched_21000.json',{'control':{k:v for k,v in c.items() if k!='per_view'},
                 'repair':{k:v for k,v in r.items() if k!='per_view'},
                 'repair_minus_control':{k:r[k]-c[k] for k in ('psnr','ssim','lpips')},
                 'note':'Same parent, training RNG, weights outside positional repair and source snapshot; all 32 held-out views'})
        if not passed:
            record('status.json',{'status':'rejected','step':target,'reason':'Position branch failed the recorded activity gate'})
            return 2
    archive=model/'model.smolgs.zip'
    run('scripts/verify_roundtrip.py','--checkpoint',current,'--archive',archive,'--output',model/'roundtrip.json')
    run('render.py','-m',archive,'-s','/root/autodl-tmp/data/tandt/truck','-r','1','-o',model/'test/ours_35000','--save-images')
    run('metrics.py','-m',model)
    run('scripts/collect_results.py','--output-root',out)
    run('scripts/compare_tnt_db.py','--output-root',out)
    record('status.json',{'status':'complete','step':35000})
    return 0


if __name__=='__main__':
    try:
        code=main()
    except Exception:
        import traceback
        traceback.print_exc();code=1
    sys.exit(code)
