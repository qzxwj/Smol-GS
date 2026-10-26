#!/usr/bin/env python3
"""Instrument native training to locate positional-MLP collapse; no automatic repair."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
import train
from scene.gaussian_model import SmolGS
from utils.graphics_utils import positional_bits, quantize_cells


def initialize_octree(module, seed=1729):
    # ReLU-appropriate initialization; no architecture/lr change. Isolate its RNG
    # so all other network weights and the initial splat features remain equal.
    device = next(module.parameters()).device
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for layer in (module[0], module[2]):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(module[4].weight)
        nn.init.zeros_(module[4].bias)


@torch.no_grad()
def activity(model, count=16384):
    ids = torch.linspace(0, len(model.xyz)-1, min(count, len(model.xyz)), device=model.xyz.device).long()
    cells, _ = quantize_cells(model.xyz[ids], model.bounds_min, model.bounds_max, model.config.octree_depth)
    bits = positional_bits(cells, model.config.octree_depth)
    first = model.octree_mlp[0](model.position_inputs(cells))
    second = model.octree_mlp[2](model.octree_mlp.hidden_input(first))
    embedding = model.octree_mlp[4](model.octree_mlp[3](second))
    return {'sample_points':len(ids), 'first_positive_fraction':float((first>0).float().mean()),
            'second_positive_fraction':float((second>0).float().mean()),
            'second_max':float(second.max()),
            'embedding_range_max':float((embedding.amax(0)-embedding.amin(0)).amax()),
            'hidden_bias_means':[float(model.octree_mlp[i].bias.mean()) for i in (0,2)],
            'gradient_norms':{name:float(param.grad.norm()) if param.grad is not None else None
                              for name,param in model.octree_mlp.named_parameters()}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--initialization', choices=['torch','kaiming'], default='torch')
    parser.add_argument('--activity-every', type=int, default=50)
    parser.add_argument('--hidden-bias', type=float, default=0.0)
    parser.add_argument('--activation', choices=['relu','leaky'], default='relu')
    parser.add_argument('--feature-std', type=float, default=0.01)
    parser.add_argument('--octree-eps', type=float, default=1e-15)
    parser.add_argument('--input-center', type=float, default=0.0)
    options,remaining=parser.parse_known_args()
    output=Path(remaining[remaining.index('-m')+1]);output.mkdir(parents=True,exist_ok=True)
    original_init=SmolGS.__init__;original_make=SmolGS.make_optimizer;original_update=SmolGS.update_learning_rates
    original_initialize=SmolGS.initialize
    current={'step':0}
    def construct(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        if options.initialization=='kaiming':
            initialize_octree(self.octree_mlp)
            self.config.octree_init = "kaiming"
            self.config.octree_init_seed = 1729
            with torch.no_grad():
                self.octree_mlp[0].bias.fill_(options.hidden_bias)
                self.octree_mlp[2].bias.fill_(options.hidden_bias)
        if options.input_center != self.config.octree_input_center:
            with torch.no_grad():
                self.octree_mlp[0].bias.add_((options.input_center-self.config.octree_input_center)*self.octree_mlp[0].weight.sum(-1))
            self.position_input_center.fill_(options.input_center)
            self.config.octree_input_center = options.input_center
        if options.activation=='leaky':
            self.config.octree_activation = 'leaky'
            self.octree_mlp[1]=nn.LeakyReLU(0.01)
            self.octree_mlp[3]=nn.LeakyReLU(0.01)
    def initialize(self,points):
        original_initialize(self,points)
        with torch.no_grad():
            self.features.mul_(options.feature_std/self.config.feature_init_std)
        self.config.feature_init_std = options.feature_std
    def update(self,optimizer,step):
        current['step']=step
        return original_update(self,optimizer,step)
    def make(self):
        optimizer=original_make(self);original_step=optimizer.step
        for group in optimizer.param_groups:
            if group['name']=='octree_mlp':
                group['eps']=options.octree_eps
        def step(*args,**kwargs):
            number=current['step']
            selected=number in (1,2,5,10,20) or number%options.activity_every==0
            before=activity(self) if selected else None
            result=original_step(*args,**kwargs)
            if selected:
                row={'step':number,'initialization':options.initialization,'hidden_bias':options.hidden_bias,'activation':options.activation,'feature_std':options.feature_std,'octree_eps':options.octree_eps,'input_center':options.input_center,'before':before,'after':activity(self)}
                with (output/'octree_activity.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                print('[Octree activity] '+json.dumps(row),flush=True)
            return result
        optimizer.step=step
        return optimizer
    SmolGS.__init__=construct;SmolGS.make_optimizer=make;SmolGS.update_learning_rates=update;SmolGS.initialize=initialize
    (output/'diagnostic_options.json').write_text(json.dumps(vars(options),indent=2)+'\n')
    train.main(remaining)

if __name__=='__main__':
    main()
