"""Training-view opacity scores and deterministic one-Gaussian-per-cell selection."""
import random
import numpy as np
import torch


def select_coordinate_survivors(codes, scores=None):
    """Return Morton-ordered representatives; ties use the smallest original index."""
    order = torch.arange(len(codes), device=codes.device)
    if scores is not None:
        if (scores.shape != codes.shape or scores.device != codes.device
                or not scores.is_floating_point() or not torch.isfinite(scores).all()
                or (scores < 0).any()):
            raise ValueError('Coordinate scores must be finite nonnegative floats, one per point on its device')
        order = torch.argsort(scores, descending=True, stable=True)
    order = order[torch.argsort(codes[order], stable=True)]
    sorted_codes = codes[order]
    keep = torch.ones_like(sorted_codes, dtype=torch.bool)
    keep[1:] = sorted_codes[1:] != sorted_codes[:-1]
    return order[keep]


@torch.no_grad()
def score_training_opacity(model, train_cameras, step, renderer='cuda', progress=True):
    """Mean opacity*1[radii>0] over ALL supplied training views, without reading GT.

    radii>0 is the renderer's projection prefilter, not an occlusion or contribution
    test. Evaluate before snapping coordinates; use inference feature quantization.
    Restore module modes, RNGs and ECVQ caches so scoring cannot alter continuation.
    """
    from gaussian_renderer import render
    if not train_cameras or len({c.name for c in train_cameras}) != len(train_cameras):
        raise ValueError('Opacity scoring needs a nonempty list of distinct training cameras')
    if bool(model.coordinate_frozen):
        raise ValueError('Score opacity before freezing coordinates')
    device = model.xyz.device
    total = torch.zeros(len(model.xyz), dtype=torch.float64, device=device)
    seen = torch.zeros(len(model.xyz), dtype=torch.long, device=device)
    modes = [(module, module.training) for module in model.modules()]
    caches = [(buffer, buffer.clone()) for name, buffer in model.named_buffers()
              if name.endswith('_index_cache')]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            model.eval()
            for number, camera in enumerate(train_cameras, 1):
                package = render(model, camera.to(device), step, renderer)
                opacity = package['attributes']['opacity'].reshape(-1)
                visible = package['visible']
                if (opacity.shape != total.shape or visible.shape != total.shape
                        or visible.dtype != torch.bool or not torch.isfinite(opacity).all()
                        or (opacity < 0).any() or (opacity > 1).any()):
                    raise ValueError('Invalid opacity/visibility returned by renderer')
                total.add_(torch.where(visible, opacity.double(), 0.))
                seen.add_(visible.long())
                del package
                if progress and (number % 25 == 0 or number == len(train_cameras)):
                    print(f'[Opacity scoring] {number}/{len(train_cameras)} training views', flush=True)
    finally:
        for buffer, saved in caches:
            buffer.copy_(saved)
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    return total/len(train_cameras), seen
