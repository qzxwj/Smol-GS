# Historical float-grid export utility; unused by the active binary encoder/codec.
"""LocoGS grid export: distribution clipping, signed log1p, and 6-bit codes.

Matches LocoGS/utils/quantization_utils.py for finite, nonconstant arrays. The
constant case is defined separately to avoid its division by zero. Operations
use copies so encoding never modifies training/checkpoint parameters.
"""
import io
import numpy as np
import torch


def pack_loco_grid(value):
    data = value.detach().float().cpu().numpy().copy()
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError('Grid parameters must be nonempty and finite')
    original = data.copy()
    # LocoGS: d = 3 + 3*(bit-1)/15; bit=6 gives four standard deviations.
    mean, std = data.mean(), data.std()
    data = data.clip(mean-4*std, mean+4*std)
    positive, negative = data > 0, data < 0
    data[positive] = np.log1p(data[positive])
    data[negative] = -np.log1p(-data[negative])
    minimum, maximum = data.min(), data.max()
    scale = 63/(maximum-minimum) if maximum != minimum else 1.
    codes = np.clip(np.round((data-minimum)*scale), 0, 63).astype(np.uint8)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=codes, scale=scale, min=minimum)
    payload = buffer.getvalue()
    restored = unpack_loco_grid(payload, tuple(value.shape))
    error = np.abs(restored.numpy()-original)
    return payload, {'bits': 6, 'transform': 'signed-log1p', 'clip_sigma': 4,
                     'shape': list(value.shape), 'count': value.numel(),
                     'max_abs_error': float(error.max()), 'mean_abs_error': float(error.mean())}


def unpack_loco_grid(payload, shape):
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        codes, scale, minimum = archive['data'], archive['scale'], archive['min']
        if (codes.dtype != np.uint8 or tuple(codes.shape) != tuple(shape) or codes.max(initial=0) > 63
                or scale.shape != () or minimum.shape != () or not np.isfinite(scale)
                or not np.isfinite(minimum) or scale <= 0):
            raise ValueError('Invalid LocoGS 6-bit grid stream')
        data = codes.astype(np.float32)/scale+minimum
    positive, negative = data > 0, data < 0
    data[positive] = np.expm1(data[positive])
    data[negative] = -np.expm1(-data[negative])
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite dequantized grid')
    return torch.from_numpy(data.astype(np.float32, copy=False))
