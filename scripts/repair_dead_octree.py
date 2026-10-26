#!/usr/bin/env python3
"""Function-preserving diagnostic repair of an all-point dead ReLU positional MLP."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from utils.checkpoint import load_checkpoint,save_checkpoint
from scene.dataset_readers import load_scene
from gaussian_renderer import render
from utils.graphics_utils import quantize_cells,positional_bits


@torch.no_grad()
def repair(model,optimizer,seed=1729,conditioning='none'):
    if conditioning not in ('none','current_points'):
        raise ValueError('Unknown conditioning mode')
    if model.config.octree_activation!='relu':
        raise ValueError('Only the user-selected ReLU architecture is allowed')
    health=model.octree_health(sample_size=None)
    if health['octree_second_active_fraction']!=0:
        raise ValueError('Refusing repair: second ReLU is not dead on every current splat')
    if model.config.octree_hidden_center_mode!='none' or model.config.octree_input_center_mode!='constant' or model.config.octree_input_center!=0:
        raise ValueError('Use the original uncentered checkpoint for this controlled repair')
    bias=model.octree_mlp[4].bias.detach().clone()
    generator=torch.Generator(device=model.xyz.device).manual_seed(seed)
    changed=[]
    for i in (0,2):
        nn.init.kaiming_normal_(model.octree_mlp[i].weight,nonlinearity='relu',generator=generator)
        nn.init.zeros_(model.octree_mlp[i].bias)
        changed += [model.octree_mlp[i].weight,model.octree_mlp[i].bias]
    if conditioning=='current_points':
        # Fixed centers from the actual 20k cloud, without using test images.
        # Zero output weights below preserve the old constant function exactly.
        bit_sum=model.xyz.new_zeros(3*model.config.octree_depth,dtype=torch.float64)
        for start in range(0,len(model.xyz),65536):
            cells,_=quantize_cells(model.xyz[start:start+65536],model.bounds_min,model.bounds_max,model.config.octree_depth)
            bit_sum+=positional_bits(cells,model.config.octree_depth,model.xyz.dtype).sum(0,dtype=torch.float64)
        center=(bit_sum/len(model.xyz)).to(model.xyz.dtype)
        model.position_input_center.copy_(center)
        model.config.octree_input_center_mode='checkpoint'
        model.config.octree_input_mean=tuple(center.cpu().tolist())
        hidden_sum=model.xyz.new_zeros(model.config.octree_mlp_width,dtype=torch.float64)
        for start in range(0,len(model.xyz),65536):
            cells,_=quantize_cells(model.xyz[start:start+65536],model.bounds_min,model.bounds_max,model.config.octree_depth)
            hidden_sum+=model.octree_mlp[1](model.octree_mlp[0](model.position_inputs(cells))).sum(0,dtype=torch.float64)
        hidden_center=(hidden_sum/len(model.xyz)).to(model.xyz.dtype)
        model.octree_mlp.hidden_center.copy_(hidden_center)
        model.octree_mlp.use_hidden_center=True
        model.config.octree_hidden_center_mode='checkpoint'
        model.config.octree_hidden_center=tuple(hidden_center.cpu().tolist())
        model.config.__post_init__()
    nn.init.zeros_(model.octree_mlp[4].weight)
    changed.append(model.octree_mlp[4].weight)
    for parameter in changed:
        parameter.grad=None
        if optimizer is not None:optimizer.state.pop(parameter,None)
    assert torch.equal(model.octree_mlp[4].bias,bias)
    after=model.octree_health(sample_size=None)
    if after['octree_embedding_range']!=0 or after['octree_hidden_nonzero_point_fraction']==0:
        raise AssertionError('Repair did not preserve the constant output and revive the hidden path')
    return {'seed':seed,'conditioning':conditioning,'before':health,'after':after,'reset_parameter_tensors':5,
            'preserved_output_bias':bias.tolist(),
            'optimizer_note':'Reset Adam state only for five reinitialized tensors; preserve the existing output-bias state and every other optimizer group',
            'scope':'Diagnostic one-time checkpoint repair; not an author-confirmed training step'}


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state().clone(),
            'cuda':[x.clone() for x in torch.cuda.get_rng_state_all()]}


def same_rng(a,b):
    return (a['python']==b['python'] and a['numpy'][0]==b['numpy'][0] and
            np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:] and
            torch.equal(a['cpu'],b['cpu']) and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda'])))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=1729)
    p.add_argument('--conditioning',choices=['none','current_points'],default='none')
    args=p.parse_args();torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,opt,step,metadata=load_checkpoint(args.checkpoint,'cuda',resume=True)
    assert step==20000 and model.config.lambda_opacity==3e-7 and model.config.lambda_rate==5e-4
    model.eval();data=metadata['data'];_,cameras=load_scene(data['source_path'],data['images'],data['resolution'],data['holdout'],data['white_background'])
    selected=[cameras[i] for i in (0,len(cameras)//2,len(cameras)-1)]
    original={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('octree_mlp.')}
    before_rng=rng_state();baseline=[]
    with torch.no_grad():
        for c in selected:
            package=render(model,c.to('cuda'),step,'cuda')
            baseline.append({'image':package['image'].clone(),
                'attributes':{k:package['attributes'][k].clone() for k in ('xyz','color','opacity','rotation','scales')}})
        report=repair(model,opt,args.seed,args.conditioning)
        errors=[]
        for c,old in zip(selected,baseline):
            package=render(model,c.to('cuda'),step,'cuda')
            diffs={k:float((package['attributes'][k]-old['attributes'][k]).abs().max()) for k in old['attributes']}
            image_diff=float((package['image']-old['image']).abs().max())
            assert all(value==0 for value in diffs.values()) and image_diff==0,(diffs,image_diff)
            errors.append({'camera':c.name,'attribute_max_abs':diffs,'image_max_abs':image_diff})
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
    assert same_rng(before_rng,rng_state()),'Diagnostic changed training RNG'
    report.update({'parent':str(Path(args.checkpoint).resolve()),'parent_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                   'step':step,'render_checks':errors,'nonpositional_state_unchanged':True,'rng_unchanged':True})
    metadata={**metadata,'positional_checkpoint_repair':report}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    save_checkpoint(out,model,opt,step,metadata)
    out.with_suffix('.repair.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
