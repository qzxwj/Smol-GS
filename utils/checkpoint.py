"""Checkpoints include model, Adam moments, stage, and all RNG states."""
from pathlib import Path
import random
import json
import numpy as np
import torch
from arguments import Config
from utils.formats import CHECKPOINT_FORMAT
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
    payload = {'format': CHECKPOINT_FORMAT, 'geometry_mode': model.config.geometry_mode,
               'geometry_quantization': model.config.geometry_quantization,
               'hash_model': 'hac-binary-mixed-detached-v1', 'config': model.config.to_dict(),
               'model': model.state_dict(), 'optimizer': optimizer.state_dict() if optimizer else None,
               'step': step, 'rng': rng, 'metadata': metadata or {},
               'adc_policy': dict(model.adc_policy)}
    if model.config.geometry_quantization == 'ecvq':
        payload['optimizer_vq'] = model.optimizer_vq.state_dict() if optimizer is not None and model.optimizer_vq else None
        payload['scheduler_vq'] = model.scheduler_vq.state_dict() if optimizer is not None and model.scheduler_vq else None
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
    if (payload.get('format') != CHECKPOINT_FORMAT
            or payload.get('hash_model') != 'hac-binary-mixed-detached-v1'):
        raise ValueError(f'Unsupported checkpoint format/hash model; expected {CHECKPOINT_FORMAT}')
    config = Config.from_saved(payload['config'])
    if payload.get('geometry_mode') != config.geometry_mode:
        raise ValueError('Checkpoint geometry mode/config mismatch')
    if payload.get('geometry_quantization') != config.geometry_quantization:
        raise ValueError('Checkpoint geometry quantization/config mismatch')
    if resume and bool(payload['model']['decoded']):
        raise ValueError('Decoded archives are inference-only; resume from a training checkpoint')
    stored_policy = payload.get('adc_policy')
    if stored_policy != ADC_POLICY:
        raise ValueError('Unsupported ADC checkpoint policy; start a fresh run')
    model = SmolGS(config, len(payload['model']['xyz']), device)
    state = payload['model']
    if config.geometry_quantization == 'ecvq' and bool(state['decoded']):
        for name in ('decoded_scales', 'scale_indices', 'rot_indices'):
            setattr(model, name, state[name].to(device))
    model.load_state_dict(state)
    if config.geometry_quantization == 'ecvq':
        initialized = [bool(q.initialized) for q in model.quantizers.values()]
        if (initialized != [payload['step'] >= config.vq_start_iter]*2
                or bool(model.vq_rate_constrained) != (payload['step'] >= config.rate_constrain_iter)):
            raise ValueError('Checkpoint ECVQ stage state is inconsistent with its iteration')
    model.xyz.requires_grad_(not bool(model.coordinate_frozen))
    optimizer = model.make_optimizer() if resume else None
    if resume:
        if payload['optimizer'] is None:
            raise ValueError('This checkpoint contains no optimizer state')
        optimizer.load_state_dict(payload['optimizer'])
        if config.geometry_quantization == 'ecvq':
            if payload.get('optimizer_vq') is None or payload.get('scheduler_vq') is None:
                raise ValueError('ECVQ resume requires optimizer and scheduler states')
            model.optimizer_vq.load_state_dict(payload['optimizer_vq'])
            model.scheduler_vq.load_state_dict(payload['scheduler_vq'])
            if model.scheduler_vq.last_epoch != int(model.vq_updates):
                raise ValueError('ECVQ scheduler/update count mismatch')
        random.setstate(payload['rng']['python'])
        np.random.set_state(payload['rng']['numpy'])
        torch.set_rng_state(payload['rng']['torch'])
        if 'cuda' in payload['rng'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload['rng']['cuda'])
    return model, optimizer, payload['step'], payload['metadata']
