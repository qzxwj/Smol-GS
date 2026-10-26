import torch
import torch.nn.functional as F


def quaternion_matrix(q):
    """Quaternion order (w,x,y,z), as used by the Inria rasterizer."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)
    ], dim=-1).reshape(-1, 3, 3)


def quantize_cells(xyz, bounds_min, bounds_max, depth):
    extent = (bounds_max - bounds_min).clamp_min(1e-6)
    size = extent / (1 << depth)
    cells = torch.floor((xyz-bounds_min)/size).long().clamp(0, (1 << depth)-1)
    centers = bounds_min + (cells.to(xyz.dtype)+0.5)*size
    return cells, centers


def morton_codes(cells, depth):
    """Children 0..7 use xyz binary weights (4,2,1)."""
    result = torch.zeros(cells.shape[0], dtype=torch.long, device=cells.device)
    for bit in range(depth-1, -1, -1):
        child = (((cells >> bit) & 1) * cells.new_tensor([4, 2, 1])).sum(-1)
        result = (result << 3) | child
    return result


def positional_bits(cells, depth, dtype=torch.float32):
    shifts = torch.arange(depth-1, -1, -1, device=cells.device)
    return ((cells[:, None, :] >> shifts[None, :, None]) & 1).reshape(-1, depth*3).to(dtype)
