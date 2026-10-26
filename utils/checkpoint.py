"""Checkpoints include model, Adam moments, stage, and all RNG states."""
from pathlib import Path
import random
import json
import numpy as np
import torch
from arguments import Config
from scene.gaussian_model import SmolGS


def load_torch(path, map_location='cpu'):
    # Explicit weights_only=False is needed because RNG checkpoints contain numpy state.
    # Only load checkpoints produced by this project or otherwise trusted sources.
    return torch.load(path, map_location=map_location, weights_only=False)


def save_checkpoint(path, model, optimizer, step, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        rng['cuda'] = torch.cuda.get_rng_state_all()
    payload = {'format': 'smolgs-checkpoint-v1', 'config': model.config.to_dict(),
               'model': model.state_dict(), 'optimizer': optimizer.state_dict() if optimizer else None,
               'step': step, 'rng': rng, 'metadata': metadata or {}}
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, str(temporary))
    temporary.replace(path)
    sidecar = Path(str(path)+'.json')
    sidecar_tmp = Path(str(sidecar)+'.tmp')
    sidecar_tmp.write_text(json.dumps({'step': step, 'iterations': model.config.iterations,
                                      'splats': len(model.xyz), 'format': payload['format']}, indent=2)+'\n')
    sidecar_tmp.replace(sidecar)


def load_checkpoint(path, device='cpu', resume=False):
    payload = load_torch(path, 'cpu')
    if payload.get('format') != 'smolgs-checkpoint-v1':
        raise ValueError('Unsupported checkpoint format')
    config = Config(**payload['config'])
    model = SmolGS(config, len(payload['model']['xyz']), device)
    state = payload['model']
    if 'scene_extent' not in state:
        if resume:
            raise ValueError('Legacy ADC checkpoint has no fixed camera extent; use it for diagnosis/rendering, '
                             'and start a fresh run for the size-aware ADC policy')
        # Old results remain readable without silently changing their training history.
        state = {**state, 'scene_extent': model.scene_extent}
    model.load_state_dict(state)
    model.xyz.requires_grad_(not bool(model.coordinate_frozen))
    optimizer = model.make_optimizer() if resume else None
    if resume:
        if payload['optimizer'] is None:
            raise ValueError('This checkpoint contains no optimizer state')
        optimizer.load_state_dict(payload['optimizer'])
        random.setstate(payload['rng']['python'])
        np.random.set_state(payload['rng']['numpy'])
        torch.set_rng_state(payload['rng']['torch'])
        if 'cuda' in payload['rng'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload['rng']['cuda'])
    return model, optimizer, payload['step'], payload['metadata']
