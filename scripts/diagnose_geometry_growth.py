#!/usr/bin/env python3
"""Read-only geometry/scale statistics from saved checkpoints; no optimization."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial import cKDTree
import torch
from scene.dataset_readers import load_points,load_scene
from utils.checkpoint import load_torch
from utils.graphics_utils import quantize_cells

root=Path(__file__).resolve().parents[1]
run=root/'outputs/truck_screen_base_20261021'
base=run/'tandt/truck/base'
torch.set_num_threads(8)
train,test=load_scene(root.parent/'data/tandt/truck','images',1,8)
camera_centers=np.stack([c.center.numpy() for c in train+test])
center=camera_centers.mean(0)
radius=float(np.linalg.norm(camera_centers-center,axis=1).max()*1.1)
levels=[0,.5,.9,.99,.999,1]

def summarize(xyz,scales):
    xyz=np.asarray(xyz);scales=np.asarray(scales)
    distances=np.linalg.norm(xyz-center,axis=1)
    largest=scales.max(1)
    extreme=np.argsort(distances)[-5:][::-1]
    return {'count':len(xyz),'xyz_min':xyz.min(0).tolist(),'xyz_max':xyz.max(0).tolist(),
            'extent':np.ptp(xyz,axis=0).tolist(),'distance_quantiles':np.quantile(distances,levels).tolist(),
            'max_scale_quantiles':np.quantile(largest,levels).tolist(),
            'points_beyond_10_camera_radii':int((distances>10*radius).sum()),
            'points_beyond_100_camera_radii':int((distances>100*radius).sum()),
            'points_with_scale_over_camera_radius':int((largest>radius).sum()),
            'points_with_scale_over_100_camera_radii':int((largest>100*radius).sum()),
            'extreme_points':[{'index':int(i),'xyz':xyz[i].tolist(),'controller':scales[i].tolist()} for i in extreme]}

points=load_points(root.parent/'data/tandt/truck')
dists,_=cKDTree(points).query(points,k=4)
scales=np.sqrt(np.maximum((dists[:,1:]**2).mean(1),1e-7))[:,None].repeat(3,axis=1)
report={'camera_center':center.tolist(),'camera_radius':radius,'quantile_levels':levels,
        'initial_sfm':summarize(points,scales),'checkpoints':[]}
print('INITIAL',json.dumps(report['initial_sfm']),flush=True)
for step in [7000,15000,20000,30000,35000]:
    payload=load_torch(base/f'checkpoint_{step}.pt','cpu')
    state=payload['model']
    xyz=state['xyz'].numpy();scales=state['log_scales'].exp().numpy()
    row={'step':step,**summarize(xyz,scales)}
    if step==30000:
        lo=state['xyz'].amin(0);hi=state['xyz'].amax(0)
        row['collision_rates_by_depth']={}
        for depth in [16,18,20]:
            cells,_=quantize_cells(state['xyz'],lo,hi,depth)
            unique=len(np.unique(cells.numpy(),axis=0))
            row['collision_rates_by_depth'][str(depth)]={'unique_cells':unique,'removed':len(xyz)-unique,'fraction':1-unique/len(xyz)}
    report['checkpoints'].append(row)
    print('CHECKPOINT',json.dumps(row),flush=True)
    del payload,state
(run/'geometry_growth_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
