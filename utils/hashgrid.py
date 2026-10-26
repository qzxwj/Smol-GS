"""HAC's binary 3D + three planar hash grids; CPU and CUDA backends."""
import importlib
import itertools
import math
import torch
from torch import nn


class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * (x.abs() <= 1)


class _CudaGrid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, table, offsets, resolutions):
        backend = importlib.import_module("_gridencoder")
        x = x.contiguous()
        n, dim = x.shape
        levels, features = len(resolutions), table.shape[-1]
        out = table.new_empty(levels, n, features)
        derivative = table.new_empty(n, levels*dim*features) if x.requires_grad else None
        backend.grid_encode_forward(x, table, offsets, resolutions, out,
                                    n, dim, features, levels, 0, 128, 0.0,
                                    derivative, None, None)
        ctx.save_for_backward(x, table, offsets, resolutions)
        ctx.derivative = derivative
        return out.permute(1, 0, 2).reshape(n, -1)

    @staticmethod
    def backward(ctx, grad):
        x, table, offsets, resolutions = ctx.saved_tensors
        n, dim = x.shape
        levels, features = len(resolutions), table.shape[-1]
        grad_table = torch.zeros_like(table)
        grad_x = torch.zeros_like(x) if ctx.derivative is not None else None
        grad = grad.reshape(n, levels, features).permute(1, 0, 2).contiguous()
        importlib.import_module("_gridencoder").grid_encode_backward(
            grad, x, table, offsets, resolutions, grad_table,
            n, dim, features, levels, 0, 128, ctx.derivative, grad_x, None, None)
        return grad_x, grad_table, None, None


class HashGrid(nn.Module):
    def __init__(self, dimensions, resolutions, log2_size, features, backend="auto"):
        super().__init__()
        self.dimensions = dimensions
        self.resolution_values = tuple(resolutions)
        self.backend = backend
        sizes = [math.ceil(min(1 << log2_size, r**dimensions)/8)*8 for r in resolutions]
        self.sizes = sizes
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1]+size)
        self.offset_values = offsets
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int32))
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.int32))
        self.table = nn.Parameter(torch.empty(offsets[-1], features).uniform_(-1e-4, 1e-4))
        self.output_dim = len(resolutions)*features

    def forward(self, x):
        table = BinarySTE.apply(self.table)
        if self.backend != "torch" and x.is_cuda:
            try:
                importlib.import_module("_gridencoder")
            except ImportError:
                if self.backend == "cuda":
                    raise RuntimeError("Build submodules/gridencoder or choose hash_backend=torch")
            else:
                return _CudaGrid.apply(x, table, self.offsets, self.resolutions)
        # Match HAC: coordinate mapping x*(resolution-2)+0.5 and uint32 XOR hash.
        valid = ((x >= 0) & (x <= 1)).all(-1, keepdim=True)
        safe_x = x.clamp(0, 1)
        outputs = []
        primes = (1, 2654435761, 805459861)
        for level, (resolution, size) in enumerate(zip(self.resolution_values, self.sizes)):
            p = safe_x*(resolution-2)+0.5
            base = p.floor().long()
            frac = p-base
            out = table.new_zeros(x.shape[0], table.shape[1])
            weight_sum = x.new_zeros(x.shape[0], 1)
            for corner in itertools.product((0, 1), repeat=self.dimensions):
                c = torch.tensor(corner, device=x.device)
                grid = base+c
                weight = torch.where(c.bool(), frac, 1-frac).prod(-1, keepdim=True)
                weight = weight*((grid > 0) & (grid < resolution-1)).all(-1, keepdim=True)
                weight_sum = weight_sum+weight
                if resolution**self.dimensions > size:
                    index = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                    for d in range(self.dimensions):
                        index ^= (grid[:, d]*primes[d]) & 0xFFFFFFFF
                else:
                    index = sum(grid[:, d]*(resolution**d) for d in range(self.dimensions))
                out = out + weight*table[index.remainder(size)+self.offset_values[level]]
            outputs.append(out/weight_sum.clamp_min(1e-9)*valid)
        return torch.cat(outputs, dim=-1)


class MixedHashEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.xyz = HashGrid(3, config.resolutions_3d, config.hash_log2_3d,
                            config.hash_features, config.hash_backend)
        for name in ("xy", "xz", "yz"):
            setattr(self, name, HashGrid(2, config.resolutions_2d, config.hash_log2_2d,
                                        config.hash_features, config.hash_backend))
        self.output_dim = self.xyz.output_dim + self.xy.output_dim*3

    def forward(self, x):
        return torch.cat([self.xyz(x), self.xy(x[:, [0, 1]]),
                          self.xz(x[:, [0, 2]]), self.yz(x[:, [1, 2]])], dim=-1)
