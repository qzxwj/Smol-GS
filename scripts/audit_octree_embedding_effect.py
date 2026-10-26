#!/usr/bin/env python3
"""Same-weight intervention: replace position-dependent embeddings by their mean."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import MethodType
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from utils.checkpoint import load_checkpoint
from utils.graphics_utils import quantize_cells
from scene.dataset_readers import load_scene
from smolgs.evaluation import evaluate


@torch.no_grad()
def embedding_mean(model):
    total = model.xyz.new_zeros(model.config.octree_feature_dim,dtype=torch.float64)
    for start in range(0,len(model.xyz),65536):
        cells = (model.cells[start:start+65536] if bool(model.coordinate_frozen) else
                 quantize_cells(model.xyz[start:start+65536],model.bounds_min,model.bounds_max,
                                model.config.octree_depth)[0])
        total += model.position_embedding(cells).sum(0,dtype=torch.float64)
    return (total/len(model.xyz)).to(model.xyz.dtype)


@contextmanager
def constant_embedding(model, mean):
    present = 'position_embedding' in model.__dict__
    previous = model.__dict__.get('position_embedding')
    def constant(self,cells):
        return mean[None].expand(len(cells),-1)
    model.position_embedding = MethodType(constant,model)
    try:
        yield
    finally:
        if present:
            model.position_embedding = previous
        else:
            del model.position_embedding


@contextmanager
def fixed_embedding_table(model, cells_reference, table):
    """Override only an explicitly checked full-scene ordering, without mutation."""
    present = 'position_embedding' in model.__dict__
    previous = model.__dict__.get('position_embedding')
    def fixed(self,cells):
        if not torch.equal(cells,cells_reference):
            raise ValueError('Embedding intervention requires the recorded full-scene cell order')
        return table
    model.position_embedding=MethodType(fixed,model)
    try:
        yield
    finally:
        if present:model.position_embedding=previous
        else:del model.position_embedding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_,step,metadata=load_checkpoint(args.checkpoint,'cuda');model.eval()
    assert model.config.octree_activation=='relu', 'This audit is restricted to the user-selected ReLU route'
    data=metadata['data'];train_cameras,cameras=load_scene(data['source_path'],data['images'],data['resolution'],data['holdout'],data['white_background'])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    health=model.octree_health(sample_size=None)
    normal=evaluate(model,cameras,step,'cuda',output/'normal',True)
    mean=embedding_mean(model)
    with constant_embedding(model,mean):
        fixed=evaluate(model,cameras,step,'cuda',output/'constant_mean',True)
    # Separate near-camera variation from offsets induced by far-background
    # embeddings. Preserve all far-point embeddings exactly in this intervention.
    with torch.no_grad():
        cells = (model.cells if bool(model.coordinate_frozen) else
                 quantize_cells(model.xyz,model.bounds_min,model.bounds_max,model.config.octree_depth)[0])
        original_embedding=model.position_embedding(cells)
        camera_mean=torch.stack([c.center for c in train_cameras]).mean(0).to(model.xyz.device)
        near=(model.xyz-camera_mean).norm(dim=-1)<=2*model.scene_extent
        near_count=int(near.sum())
        near_fixed=None
        if near_count:
            table=original_embedding.clone()
            near_mean=original_embedding[near].mean(0,dtype=torch.float64).to(model.xyz.dtype)
            table[near]=near_mean
            with fixed_embedding_table(model,cells,table):
                near_fixed=evaluate(model,cameras,step,'cuda',output/'near_constant_mean',True)
    report={'step':step,'checkpoint':str(Path(args.checkpoint).resolve()),
            'scope':'Immediate same-weight intervention; not a retrained no-PE model',
            'intervention':'Replace spatial embedding with its all-splat mean; retain features, geometry, view input and every weight',
            'gpu_all_point_health':health,'mean_embedding':mean.tolist(),
            'normal':{k:v for k,v in normal.items() if k!='per_view'},
            'constant_mean':{k:v for k,v in fixed.items() if k!='per_view'},
            'normal_minus_constant':{k:normal[k]-fixed[k] for k in ('psnr','ssim','lpips')},
            'near_group_note':'Distance<=2*fixed camera extent from mean training-camera center, not a visibility or foreground mask',
            'near_group_points':near_count,
            'near_constant_mean':{k:v for k,v in near_fixed.items() if k!='per_view'} if near_fixed else None,
            'normal_minus_near_constant':{k:normal[k]-near_fixed[k] for k in ('psnr','ssim','lpips')} if near_fixed else None}
    (output/'effect.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
