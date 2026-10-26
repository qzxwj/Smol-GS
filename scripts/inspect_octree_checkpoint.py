#!/usr/bin/env python3
"""Measure point-wise positional activity, optionally near the camera cloud."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from utils.checkpoint import load_checkpoint
from utils.graphics_utils import positional_bits,quantize_cells
from scene.dataset_readers import load_scene


@torch.no_grad()
def inspect(model, camera_mean=None):
    groups={name:{'points':0,'active_points':0,'second_positive_elements':0,
                  'embedding_min':None,'embedding_max':None,'embedding_sum':None,'embedding_square_sum':None}
            for name in (('all','near_camera_mean','far_from_camera_mean') if camera_mean is not None else ('all',))}
    for start in range(0,len(model.xyz),65536):
        xyz=model.xyz[start:start+65536]
        cells=model.cells[start:start+65536] if bool(model.coordinate_frozen) else quantize_cells(xyz,model.bounds_min,model.bounds_max,model.config.octree_depth)[0]
        first=model.octree_mlp[0](model.position_inputs(cells))
        second=model.octree_mlp[2](model.octree_mlp.hidden_input(first))
        hidden=model.octree_mlp[3](second)
        embedding=model.octree_mlp[4](hidden)
        masks={'all':torch.ones(len(xyz),device=xyz.device,dtype=torch.bool)}
        if camera_mean is not None:
            near=(xyz-camera_mean).norm(dim=-1)<=2*model.scene_extent
            masks.update(near_camera_mean=near,far_from_camera_mean=~near)
        for name,mask in masks.items():
            n=int(mask.sum())
            if not n:continue
            d=groups[name];values=embedding[mask].double();pre=second[mask]
            d['points']+=n;d['active_points']+=int((hidden[mask]!=0).any(-1).sum())
            d['second_positive_elements']+=int((pre>0).sum())
            for key,value,op in [('embedding_min',values.amin(0),torch.minimum),('embedding_max',values.amax(0),torch.maximum),
                                 ('embedding_sum',values.sum(0),torch.add),('embedding_square_sum',values.square().sum(0),torch.add)]:
                d[key]=value if d[key] is None else op(d[key],value)
    for d in groups.values():
        if not d['points']:continue
        n=d['points'];d['active_point_fraction']=d['active_points']/n
        d['second_positive_fraction']=d['second_positive_elements']/(n*model.config.octree_mlp_width)
        d['embedding_range_max']=float((d['embedding_max']-d['embedding_min']).max())
        d['embedding_std_per_channel']=((d.pop('embedding_square_sum')/n-(d.pop('embedding_sum')/n).square()).clamp_min(0).sqrt()).tolist()
        for k in ('embedding_min','embedding_max'):d[k]=d[k].tolist()
    return groups


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    args=p.parse_args();torch.set_num_threads(4)
    model,_,step,metadata=load_checkpoint(args.checkpoint,args.device)
    data=metadata.get('data');camera_mean=None
    if data:
        train,_=load_scene(data['source_path'],data['images'],data['resolution'],data['holdout'],data['white_background'])
        camera_mean=torch.stack([c.center for c in train]).mean(0).to(args.device)
    report={'checkpoint':str(Path(args.checkpoint).resolve()),'step':step,'device':args.device,
            'activity_note':'active_points counts nonzero output of the second activation; for LeakyReLU negative outputs remain active. second_positive_fraction always counts preactivation >0.',
            'group_note':'near means distance <=2*fixed camera extent from the mean training-camera center; this is not a visibility or foreground classification',
            'groups':inspect(model,camera_mean)}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
