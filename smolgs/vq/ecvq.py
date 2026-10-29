"""Adapted from RDO-Gaussian vq/ecvq.py and vq/kmeans.py.

See docs/ECVQ.md and THIRD_PARTY_NOTICES.md for provenance and
edge-case changes. This implementation retains one full-vector codebook per
attribute, Euclidean (not squared) distances, and the reference probability floor.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ECVQ(nn.Module):
    def __init__(self, x_dim, cb_size, cb_dim, lmbda, patch_size=16384):
        super().__init__()
        if x_dim != cb_dim or cb_dim < 1 or cb_size < 1 or patch_size < 1:
            raise ValueError('Geometry ECVQ requires one full-vector codebook and positive sizes')
        if not math.isfinite(lmbda) or lmbda <= 0:
            raise ValueError('ECVQ lambda must be finite and positive')
        self.cb_dim, self.cb_size = cb_dim, cb_size
        self.lmbda, self.patch_size = lmbda, patch_size
        self.codebook = nn.Parameter(torch.empty(1, cb_size, cb_dim).normal_(0, 1/math.sqrt(cb_dim)))
        self.logits = nn.Parameter(torch.zeros(1, cb_size))
        self.register_buffer('initialized', torch.tensor(False))

    def log_pmf(self):
        # RDO's Softmax.log_pmf clamps during autograd. Apply its same floor in
        # evaluation too, so assignment does not depend on torch.no_grad().
        return F.log_softmax(self.logits, dim=-1).clamp_min(math.log(1e-9))

    def _validate(self, x):
        if x.ndim != 2 or x.shape[1] != self.cb_dim:
            raise ValueError(f'Expected ECVQ input [N,{self.cb_dim}]')

    def forward(self, x, rate_constrain=False, index_cache=None):
        self._validate(x)
        if not bool(self.initialized):
            raise RuntimeError('Initialize ECVQ with K-means before quantization')
        log2_cost = self.log_pmf()/(-math.log(2))
        indexes = []
        # argmin has no derivative. Searching without a graph saves the large
        # distance matrix; selected codewords and rate costs stay differentiable.
        if index_cache is None:
            with torch.no_grad():
                for patch in x.split(self.patch_size):
                    dist = torch.cdist(patch.unsqueeze(0), self.codebook, p=2)[0]
                    if rate_constrain:
                        dist = dist+log2_cost[0]/self.lmbda
                    indexes.append(dist.argmin(dim=-1))
            index = torch.cat(indexes) if indexes else torch.empty(0, dtype=torch.long, device=x.device)
        else:
            if (index_cache.shape != (len(x), 1) or index_cache.dtype != torch.long
                    or index_cache.device != x.device
                    or (index_cache < 0).any() or (index_cache >= self.cb_size).any()):
                raise ValueError('Invalid ECVQ index cache')
            index = index_cache[:, 0]
        # Explicit dimensions avoid the reference's squeeze() scalar for N=1.
        x_hat = self.codebook[0].index_select(0, index)
        bits = log2_cost[0].index_select(0, index).sum()
        return {'x_hat': x_hat, 'bits': bits, 'x_index': index[:, None]}

    @torch.no_grad()
    def kmeans_initialize(self, x):
        self._validate(x)
        if not len(x) or not torch.isfinite(x).all():
            raise ValueError('K-means needs nonempty finite vectors')
        # Same sample_vectors and ten Lloyd steps as RDO vq/kmeans.py, with
        # native torch reshapes in place of einops (no extra runtime dependency).
        if len(x) >= self.cb_size:
            selected = torch.randperm(len(x), device=x.device)[:self.cb_size]
        else:
            selected = torch.randint(len(x), (self.cb_size,), device=x.device)
        means = x[selected].clone()
        for _ in range(10):
            buckets = torch.cat([torch.cdist(patch.unsqueeze(0), means.unsqueeze(0), 2)[0].argmin(-1)
                                 for patch in x.split(self.patch_size)])
            bins = torch.bincount(buckets, minlength=self.cb_size)
            sums = torch.zeros_like(means)
            sums.scatter_add_(0, buckets[:, None].expand(-1, self.cb_dim), x)
            updated = sums/bins.clamp_min(1)[:, None]
            means = torch.where((bins == 0)[:, None], means, updated)
        self.codebook.copy_(means[None])
        # The reference uses L2-normalized counts then log. Preserve that
        # normalization, but make empty clusters finite rather than log(0).
        probabilities = F.normalize(bins.float()[None], dim=-1).clamp_min(1e-9)
        self.logits.copy_(probabilities.log())
        self.initialized.fill_(True)

    @torch.no_grad()
    def reactivate_codeword(self, prob_threshold=1e-5):
        log_pmf = self.log_pmf()[0]
        live = log_pmf > math.log(prob_threshold)
        if not live.any():
            live[log_pmf.argmax()] = True
        dead = ~live
        count = int(dead.sum())
        if count:
            selected = log_pmf[live].exp().multinomial(count, replacement=True)
            # Match the reference's CPU noise generation before device transfer.
            noise = torch.normal(0., 1e-4, size=(count, self.cb_dim)).to(self.codebook.device)
            self.codebook[0, dead] = self.codebook[0, live][selected]+noise
            self.logits[0, dead] = self.logits[0, live][selected]
        return count
