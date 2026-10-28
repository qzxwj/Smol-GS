"""Checkpoints include model, Adam moments, stage, and all RNG states."""
from pathlib import Path
import random
import json
import numpy as np
import torch
from arguments import Config
from scene.gaussian_model import SmolGS, ADC_POLICY


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
    checkpoint_format = ('smolgs-checkpoint-v4-explicit' if model.config.geometry_mode == 'explicit'
                         else 'smolgs-checkpoint-v3')
    payload = {'format': checkpoint_format, 'geometry_mode': model.config.geometry_mode,
               'hash_model': 'hac-binary-mixed-detached-v1', 'config': model.config.to_dict(),
               'model': model.state_dict(), 'optimizer': optimizer.state_dict() if optimizer else None,
               'step': step, 'rng': rng, 'metadata': metadata or {},
               'adc_policy': dict(model.adc_policy)}
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
    if (payload.get('format') not in ('smolgs-checkpoint-v3', 'smolgs-checkpoint-v4-explicit')
            or payload.get('hash_model') != 'hac-binary-mixed-detached-v1'):
        raise ValueError('Unsupported checkpoint format/hash model; expected v3 neural or v4-explicit')
    config = Config.from_saved(payload['config'])
    expected_mode = 'explicit' if payload['format'] == 'smolgs-checkpoint-v4-explicit' else 'neural'
    if config.geometry_mode != expected_mode or payload.get('geometry_mode', 'neural') != expected_mode:
        raise ValueError('Checkpoint geometry mode/format mismatch; start a fresh run to change geometry')
    if resume and bool(payload['model']['decoded']):
        raise ValueError('Decoded archives are inference-only; resume from a training checkpoint')
    stored_policy = payload.get('adc_policy')
    if stored_policy != ADC_POLICY:
        raise ValueError('Unsupported ADC checkpoint policy; start a fresh run')
    model = SmolGS(config, len(payload['model']['xyz']), device)
    state = payload['model']
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
