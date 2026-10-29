"""Coordinate coding independent of neural positional encoding.

Optional quantile bounds define a main region; exterior points are retained in
a separate global grid. Region membership is implicit in stream order at decode.
"""
import numpy as np
import torch
from utils.graphics_utils import morton_codes, quantize_cells
from utils.octree import encode_octree, decode_octree
from smolgs.coordinate_selection import select_coordinate_survivors


def reconstruct(cells, regions, lower, upper, depth):
    return lower[regions]+(cells.to(torch.float32)+.5)*(
        upper[regions]-lower[regions]).clamp_min(1e-6)/(1 << depth)


@torch.no_grad()
def freeze_grid(model, optimizer=None, scores=None):
    xyz, depth = model.xyz, model.config.coordinate_depth
    if not len(xyz) or not torch.isfinite(xyz).all():
        raise ValueError('Coordinate coding requires nonempty finite positions')
    lo, hi = xyz.amin(0), xyz.amax(0)
    hi = torch.maximum(hi, lo+1e-6)
    tail = model.config.coordinate_tail_fraction
    main = torch.quantile(xyz, xyz.new_tensor([tail, 1-tail]), dim=0) if tail else torch.stack([lo, hi])
    lower = torch.stack([main[0], lo])
    upper = torch.maximum(torch.stack([main[1], hi]), lower+1e-6)
    regions = (~((xyz >= main[0]) & (xyz <= main[1])).all(-1)).long() if tail else torch.zeros(len(xyz), device=xyz.device, dtype=torch.long)
    size = (upper[regions]-lower[regions])/(1 << depth)
    cells = torch.floor((xyz-lower[regions])/size).long().clamp(0, (1 << depth)-1)
    centers = reconstruct(cells, regions, lower, upper, depth)
    # Each depth <=20 Morton code occupies at most 60 bits.
    codes = morton_codes(cells, depth)+regions*(1 << 60)
    indices = select_coordinate_survivors(codes, scores)
    removed = len(xyz)-len(indices)
    if removed/len(xyz) > model.config.coordinate_max_collision_fraction:
        raise RuntimeError(f'Coordinate collision fraction {removed/len(xyz):.2%} is above the collision limit')
    model.coordinate_cells = cells
    model.coordinate_regions = regions
    model.coordinate_bounds_min.copy_(lower)
    model.coordinate_bounds_max.copy_(upper)
    model.replace_topology(indices, optimizer, centers[indices])
    # Preserve the positional network's depth and coordinate normalization.
    model.cells, _ = quantize_cells(model.xyz, model.bounds_min, model.bounds_max, model.config.octree_depth)
    model.coordinate_frozen.fill_(True)
    model.xyz.requires_grad_(False)
    return removed


def encode_coordinates(model):
    """Return files, layout, canonical order and decoder-identical xyz."""
    depth = model.config.coordinate_depth
    files, descriptions, orders = {}, [], []
    for region in range(2 if model.config.coordinate_tail_fraction else 1):
        indices = torch.where(model.coordinate_regions == region)[0]
        name = f'coordinates/region_{region}.oct'
        if len(indices):
            blob, order = encode_octree(model.coordinate_cells[indices].cpu().numpy(), depth)
            files[name] = blob
            indices = indices[torch.as_tensor(order.astype(np.int64), device=indices.device)]
        descriptions.append({'region': region, 'count': len(indices), 'file': name if len(indices) else None})
        orders.append(indices)
    order = torch.cat(orders)
    if len(order) != len(model.xyz):
        raise ValueError('Invalid coordinate region membership')
    layout = {'version': 1, 'depth': depth, 'regions': descriptions,
              'lower': model.coordinate_bounds_min.tolist(), 'upper': model.coordinate_bounds_max.tolist()}
    xyz = reconstruct(model.coordinate_cells[order], model.coordinate_regions[order],
                      model.coordinate_bounds_min, model.coordinate_bounds_max, depth)
    if not torch.equal(xyz, model.xyz[order]):
        raise ValueError('Frozen positions differ from coordinate bitstream reconstruction')
    return files, layout, order, xyz


def decode_coordinates(model, files, layout, count):
    if not isinstance(layout, dict):
        raise ValueError('Missing independent coordinate layout')
    depth = model.config.coordinate_depth
    expected_regions = 2 if model.config.coordinate_tail_fraction else 1
    if layout.get('version') != 1 or layout.get('depth') != depth or len(layout['regions']) != expected_regions:
        raise ValueError('Unsupported coordinate layout')
    lower = torch.tensor(layout['lower'], device=model.xyz.device, dtype=torch.float32)
    upper = torch.tensor(layout['upper'], device=model.xyz.device, dtype=torch.float32)
    if lower.shape != (2,3) or upper.shape != (2,3) or not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or not (upper > lower).all():
        raise ValueError('Invalid coordinate bounds')
    cells, regions = [], []
    for region, description in enumerate(layout['regions']):
        if description['region'] != region or not isinstance(description['count'], int) or description['count'] < 0:
            raise ValueError('Invalid coordinate region')
        if not description['count']:
            if description['file'] is not None:
                raise ValueError('Empty coordinate region has a payload')
            continue
        current, actual_depth = decode_octree(files[description['file']])
        if actual_depth != depth or len(current) != description['count']:
            raise ValueError('Coordinate region count/depth mismatch')
        cells.append(torch.as_tensor(current, device=model.xyz.device))
        regions.append(torch.full((len(current),), region, dtype=torch.long, device=model.xyz.device))
    if not cells or sum(len(c) for c in cells) != count:
        raise ValueError('Coordinate count mismatch')
    model.coordinate_cells = torch.cat(cells)
    model.coordinate_regions = torch.cat(regions)
    model.coordinate_bounds_min.copy_(lower)
    model.coordinate_bounds_max.copy_(upper)
    model.xyz.copy_(reconstruct(model.coordinate_cells, model.coordinate_regions, lower, upper, depth))
    model.cells, _ = quantize_cells(model.xyz, model.bounds_min, model.bounds_max, model.config.octree_depth)
