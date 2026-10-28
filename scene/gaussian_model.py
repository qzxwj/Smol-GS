"""Neural splats, entropy prior, topology updates and Adam state migration."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.graphics_utils import morton_codes, positional_bits, quantize_cells, quaternion_matrix
from utils.hashgrid import MixedHashEncoder


ADC_POLICY = {"gradient_space": "screen_xy", "topology_policy": "3dgs-clone-split-v1"}


def mlp(inputs, hidden, outputs, activation=None):
    layers = [nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
              nn.ReLU(), nn.Linear(hidden, outputs)]
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)




def quantize(x, step, training):
    if training:
        return x + (torch.rand_like(x)-0.5)*step
    return torch.round(x/step)*step


def gaussian_bits(x, mean, std, step, half_width=0.5):
    std = std.clamp_min(1e-6)
    # Stable tails: mirror both endpoints when the bin lies above the mean.
    lo = (x-mean-half_width*step)/std
    hi = (x-mean+half_width*step)/std
    sign = torch.where(lo+hi >= 0, -torch.ones_like(lo), torch.ones_like(lo))
    probability = (torch.erfc(-sign*hi/math.sqrt(2)) -
                   torch.erfc(-sign*lo/math.sqrt(2))).abs()*0.5
    return -torch.log2(probability.clamp_min(1e-9))


class SmolGS(nn.Module):
    def __init__(self, config, count=0, device="cpu"):
        super().__init__()
        self.config = config
        if config.appearance_backend == 'tcnn' and not str(device).startswith('cuda'):
            raise ValueError('The native appearance backend requires device=cuda')
        if config.octree_backend == 'tcnn' and not str(device).startswith('cuda'):
            raise ValueError('The native positional backend requires device=cuda')
        self.adc_policy = dict(ADC_POLICY)
        self.xyz = nn.Parameter(torch.zeros(count, 3))
        self.features = nn.Parameter(torch.zeros(count, config.feature_dim))
        self.log_scales = nn.Parameter(torch.zeros(count, 3))
        if config.geometry_mode == 'explicit':
            rotations = torch.zeros(count, 4)
            rotations[:, 0] = 1
            self.rotations = nn.Parameter(rotations)
        self.register_buffer("bounds_min", torch.zeros(3))
        self.register_buffer("bounds_max", torch.ones(3))
        self.register_buffer("hash_min", torch.zeros(3))
        self.register_buffer("hash_max", torch.ones(3))
        self.register_buffer("cells", torch.zeros(count, 3, dtype=torch.long))
        for name in ("grad_accum", "vis_count", "opacity_accum", "last_seen"):
            self.register_buffer(name, torch.zeros(count))
        self.register_buffer("coordinate_frozen", torch.tensor(False))
        self.register_buffer("decoded", torch.tensor(False))
        # Fixed camera-derived extent, never recomputed from growing splat bounds.
        self.register_buffer("scene_extent", torch.tensor(0.0))
        self.last_adc_report = {}
        self.octree_mlp = mlp(3*config.octree_depth, config.octree_mlp_width, config.octree_feature_dim)
        dim = config.feature_dim+config.octree_feature_dim+4
        self.opacity_mlp = mlp(dim, config.mlp_width, 1, nn.Tanh())
        self.color_mlp = mlp(dim, config.mlp_width, 3, nn.Sigmoid())
        # Consume the baseline heads' RNG even in explicit mode so unchanged
        # branches and feature initialization remain comparable at the same seed.
        rotation_head = mlp(dim, config.mlp_width, 4)
        scale_head = mlp(dim, config.mlp_width, 3)
        if config.geometry_mode == 'neural':
            self.rotation_mlp = rotation_head
            self.scale_mlp = scale_head
        self.hash_encoder = MixedHashEncoder(config)
        # Table A.2: 2*n_f + (3 mean_s + 3 sigma_s + 1 delta_f + 1 delta_s).
        self.entropy_mlp = mlp(self.hash_encoder.output_dim, config.mlp_width, 2*config.feature_dim+8)
        if config.geometry_mode == 'explicit':
            # Retain exactly the baseline feature mean/std/step rows, without
            # allocating another random layer or keeping any scale-prior rows.
            head = self.entropy_mlp[-1]
            rows = list(range(2*config.feature_dim))+[2*config.feature_dim+6]
            head.weight = nn.Parameter(head.weight.detach()[rows].clone())
            head.bias = nn.Parameter(head.bias.detach()[rows].clone())
            head.out_features = len(rows)
        self.to(device)
        if config.appearance_backend == 'tcnn':
            from utils.fused_mlp import FusedMLP
            # Preserve the initialization/RNG of every other branch. Only these
            # two already-constructed Torch heads are replaced in this variant.
            with torch.cuda.device(device):
                self.opacity_mlp = FusedMLP(dim, config.mlp_width, 1)
                self.color_mlp = FusedMLP(dim, config.mlp_width, 3)
        if config.octree_backend == 'tcnn':
            from utils.fused_mlp import FusedMLP
            with torch.cuda.device(device):
                self.octree_mlp = FusedMLP(3*config.octree_depth, config.octree_mlp_width,
                    config.octree_feature_dim, seed=config.octree_init_seed)

    @torch.no_grad()
    def initialize(self, points):
        points = torch.as_tensor(points, dtype=torch.float32, device=self.xyz.device)
        if len(points) < 2 or not torch.isfinite(points).all():
            raise ValueError("Initialization requires at least two finite SfM points")
        if points.is_cuda:
            from simple_knn._C import distCUDA2
            distances = distCUDA2(points.contiguous()).clamp_min(1e-7).sqrt()
        else:
            from scipy.spatial import cKDTree
            distances, _ = cKDTree(points.cpu().numpy()).query(points.cpu().numpy(), k=min(4, len(points)))
            distances = torch.as_tensor(np.sqrt((distances[:, 1:]**2).mean(1)),
                                        device=points.device, dtype=points.dtype).clamp_min(math.sqrt(1e-7))
        self.xyz = nn.Parameter(points)
        self.features = nn.Parameter(torch.randn(len(points), self.config.feature_dim, device=points.device)*self.config.feature_init_std)
        self.log_scales = nn.Parameter(distances.log()[:, None].repeat(1, 3))
        if self.config.geometry_mode == 'explicit':
            rotations = points.new_zeros(len(points), 4)
            rotations[:, 0] = 1
            self.rotations = nn.Parameter(rotations)
        self.reset_statistics(len(points))
        self.cells = torch.zeros_like(points, dtype=torch.long)
        self.refresh_bounds()
        self.hash_min.copy_(self.bounds_min)
        self.hash_max.copy_(self.bounds_max)

    def reset_statistics(self, count):
        for name in ("grad_accum", "vis_count", "opacity_accum", "last_seen"):
            setattr(self, name, self.xyz.new_zeros(count))

    @torch.no_grad()
    def set_scene_extent(self, camera_centers):
        centers = torch.as_tensor(camera_centers, device=self.xyz.device, dtype=self.xyz.dtype)
        if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) < 2 or not torch.isfinite(centers).all():
            raise ValueError("Scene extent needs finite training camera centers of shape [N,3]")
        radius = 1.1*(centers-centers.mean(0)).norm(dim=-1).amax()
        if not torch.isfinite(radius) or radius <= 0:
            raise ValueError("Training cameras must define a positive scene extent")
        self.scene_extent.copy_(radius)

    @torch.no_grad()
    def geometry_summary(self):
        return {"geometry_mode": self.config.geometry_mode,
                "bbox_extent": (self.xyz.amax(0)-self.xyz.amin(0)).tolist(),
                "controller_scale_max": float(self.log_scales.amax().exp()),
                "scene_extent": float(self.scene_extent)}

    @torch.no_grad()
    def position_activations(self, cells):
        inputs = self.position_inputs(cells)
        if self.config.octree_backend == 'tcnn':
            return self.octree_mlp.activity(inputs)
        first = self.octree_mlp[0](inputs)
        second = self.octree_mlp[2](self.octree_mlp[1](first))
        hidden = self.octree_mlp[3](second)
        return first, second, hidden, self.octree_mlp[4](hidden)

    @torch.no_grad()
    def octree_health(self, sample_size=16384):
        """Deterministic activity probe; never consumes training RNG or gradients."""
        count = min(sample_size, len(self.xyz)) if sample_size else len(self.xyz)
        indices = torch.linspace(0, len(self.xyz)-1, count, device=self.xyz.device).long()
        first_positive = second_positive = nonzero_points = 0; second_max = -float("inf")
        minima = maxima = None
        for chunk in indices.split(65536):
            if bool(self.coordinate_frozen):
                cells = self.cells[chunk]
            else:
                cells, _ = quantize_cells(self.xyz[chunk], self.bounds_min, self.bounds_max,
                                         self.config.octree_depth)
            first, second, hidden, embedding = self.position_activations(cells)
            nonzero_points += int((hidden != 0).any(-1).sum())
            lo, hi = embedding.amin(0), embedding.amax(0)
            minima = lo if minima is None else torch.minimum(minima, lo)
            maxima = hi if maxima is None else torch.maximum(maxima, hi)
            first_positive += int((first > 0).sum())
            second_positive += int((second > 0).sum())
            second_max = max(second_max, float(second.max()))
        total = count*self.config.octree_mlp_width
        gradient_norm = (self.octree_mlp.hidden_gradient_norm() if self.config.octree_backend == 'tcnn'
                         else sum(float(p.grad.norm()) for layer in (self.octree_mlp[0], self.octree_mlp[2])
                                  for p in layer.parameters() if p.grad is not None))
        return {"octree_probe_points": count, "octree_probe_all_points": count == len(self.xyz),
                "octree_activity_method": ('torch-fp16-reconstruction' if self.config.octree_backend == 'tcnn'
                                             else 'torch-layer-activations'),
                "octree_first_active_fraction": first_positive/total,
                "octree_second_active_fraction": second_positive/total,
                "octree_second_max": second_max,
                "octree_hidden_nonzero_point_fraction": nonzero_points/count,
                "octree_embedding_range": float((maxima-minima).amax()),
                "octree_hidden_gradient_norm": gradient_norm}

    @torch.no_grad()
    def refresh_bounds(self):
        lo, hi = self.xyz.amin(0), self.xyz.amax(0)
        self.bounds_min.copy_(lo)
        self.bounds_max.copy_(torch.maximum(hi, lo+1e-6))

    def prior(self, xyz):
        # LocoGS-style branch boundary: rendering still differentiates xyz,
        # while entropy context learns its tables/networks at detached locations.
        if self.config.hash_detach_coordinates:
            xyz = xyz.detach()
        normalized = (xyz-self.hash_min)/(self.hash_max-self.hash_min).clamp_min(1e-6)
        raw = self.entropy_mlp(self.hash_encoder(normalized))
        d = self.config.feature_dim
        if self.config.geometry_mode == 'explicit':
            mf, sf, qf = raw.split([d, d, 1], dim=-1)
            return (mf, F.softplus(sf)+1e-6,
                    self.config.feature_step*(1+torch.tanh(qf)).clamp_min(1e-4))
        mf, sf, ms, ss, qf, qs = raw.split([d, d, 3, 3, 1, 1], dim=-1)
        # Positive Gaussian std and HAC's bounded step adjustment.
        return (mf, F.softplus(sf)+1e-6, ms, F.softplus(ss)+1e-6,
                self.config.feature_step*(1+torch.tanh(qf)).clamp_min(1e-4),
                self.config.scale_step*(1+torch.tanh(qs)).clamp_min(1e-4))

    def position_inputs(self, cells):
        return positional_bits(cells,self.config.octree_depth,self.xyz.dtype)

    def position_embedding(self, cells):
        return self.octree_mlp(self.position_inputs(cells))

    def attributes(self, camera_center, step=0, mask=None):
        if mask is None:
            mask = torch.ones(len(self.xyz), dtype=torch.bool, device=self.xyz.device)
        xyz = self.xyz[mask]
        features = self.features[mask]
        scales = self.log_scales[mask].exp()
        rate = xyz.new_zeros(())
        compressed = step > self.config.compact_until
        if compressed and not bool(self.decoded):
            if self.config.geometry_mode == 'explicit':
                mf, sf, qf = self.prior(xyz)
            else:
                mf, sf, ms, ss, qf, qs = self.prior(xyz)
            features = quantize(features, qf, self.training)
            if self.config.geometry_mode == 'neural':
                scales = quantize(scales, qs, self.training)
            if self.training and len(xyz):
                bits = gaussian_bits(features, mf, sf, qf, self.config.rate_bin_half_width).sum(-1)
                if self.config.geometry_mode == 'neural':
                    bits = bits+gaussian_bits(scales, ms, ss, qs, self.config.rate_bin_half_width).sum(-1)
                rate = bits.mean()
        if bool(self.coordinate_frozen):
            cells = self.cells[mask]
        else:
            cells, _ = quantize_cells(xyz, self.bounds_min, self.bounds_max, self.config.octree_depth)
        embedding = self.position_embedding(cells)
        view = camera_center[None]-xyz
        distance = view.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        augmented = torch.cat([features, view/distance, distance, embedding], dim=-1)
        if self.config.appearance_backend == 'tcnn':
            opacity = self.opacity_mlp(augmented).tanh().abs()
            # Keep Smol-GS's RGB representation and its bounded color output.
            color = torch.sigmoid(self.color_mlp(augmented))
        else:
            opacity = self.opacity_mlp(augmented).abs()
            color = self.color_mlp(augmented)
        if self.config.geometry_mode == 'explicit':
            raw_rotation = self.rotations[mask]
            # A zero quaternion does not describe a rotation. Deterministically
            # render identity near zero, including after checkpoint loading.
            identity = torch.zeros_like(raw_rotation)
            identity[:, 0] = 1
            raw_rotation = torch.where(raw_rotation.norm(dim=-1, keepdim=True) > 1e-12,
                                       raw_rotation, identity)
            rotation = F.normalize(raw_rotation, dim=-1)
        else:
            rotation = F.normalize(self.rotation_mlp(augmented), dim=-1)
            # Noise may cross zero; physical Gaussian scales must stay positive.
            scales = scales.clamp_min(1e-7)*torch.sigmoid(self.scale_mlp(augmented))
        return {"xyz": xyz, "color": color, "opacity": opacity,
                "scales": scales, "rotation": rotation, "rate": rate, "mask": mask}

    def make_optimizer(self):
        groups = []
        for name, (initial, _) in self.config.optimizer_rates().items():
            value = getattr(self, name)
            params = list(value.parameters()) if isinstance(value, nn.Module) else [value]
            groups.append({"params": params, "lr": initial, "name": name})
        return torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    def update_learning_rates(self, optimizer, step):
        t = min(max(step/self.config.lr_max_steps, 0), 1)
        rates = self.config.optimizer_rates()
        for group in optimizer.param_groups:
            a, b = rates[group["name"]]
            group["lr"] = math.exp((1-t)*math.log(a)+t*math.log(b))
            if group["name"] == "xyz" and bool(self.coordinate_frozen):
                group["lr"] = 0.0

    @torch.no_grad()
    def replace_topology(self, indices, optimizer=None, xyz=None, new_start=None, log_scales=None):
        """Reindex splats; carry Adam moments for survivors, zero them for clones."""
        for name in self.point_parameter_names:
            old = getattr(self, name)
            data = old[indices].clone()
            if name == "xyz" and xyz is not None:
                data = xyz
            if name == "log_scales" and log_scales is not None:
                data = log_scales
            new = nn.Parameter(data, requires_grad=old.requires_grad)
            setattr(self, name, new)
            if optimizer is not None:
                for group in optimizer.param_groups:
                    if group["name"] == name:
                        group["params"] = [new]
                state = optimizer.state.pop(old, None)
                if state:
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key in state:
                            state[key] = state[key][indices].clone()
                            if new_start is not None:
                                state[key][new_start:] = 0
                    optimizer.state[new] = state
        for name in ("cells", "grad_accum", "vis_count", "opacity_accum", "last_seen"):
            setattr(self, name, getattr(self, name)[indices].clone())

    @torch.no_grad()
    def collect_statistics(self, attributes, visible, step, viewspace_points=None):
        if not self.config.warmup_until < step <= self.config.compact_until:
            return
        indices = attributes["mask"].nonzero(as_tuple=True)[0]
        seen = indices[visible]
        gradient_norm = None
        if step <= self.config.densify_until:
            if viewspace_points is None or viewspace_points.grad is None:
                raise RuntimeError("Screen-space ADC needs viewspace_points.grad after loss.backward()")
            if viewspace_points.shape[0] != len(indices) or viewspace_points.shape[-1] < 2:
                raise ValueError("Projected positions must align with rendered splats and have XY components")
            gradient_norm = viewspace_points.grad[visible, :2].norm(dim=-1)
        self.vis_count[seen] += 1
        self.opacity_accum[seen] += attributes["opacity"].detach()[visible, 0]
        self.last_seen[seen] = step
        if gradient_norm is not None:
            self.grad_accum[seen] += gradient_norm

    @torch.no_grad()
    def density_control(self, optimizer, camera_center, step, max_splats=None):
        """Clone small splats; replace large parents with two smaller children.

        Sizes and sampling covariance use rendered axes. In explicit mode
        these are camera independent; children inherit the parent's rotation.
        Log scales shrink by log(0.8*2) for split children in both modes.
        """
        self.last_adc_report = {}
        if not self.config.warmup_until < step <= self.config.compact_until:
            return 0, 0
        if self.adc_policy != ADC_POLICY:
            raise ValueError("Legacy ADC model cannot use the new topology policy; start a fresh run")
        if max_splats is not None and max_splats < 1:
            raise ValueError("max_splats must be positive")
        for value in (*(getattr(self, name) for name in self.point_parameter_names), self.grad_accum,
                      self.vis_count, self.opacity_accum, self.last_seen):
            if not torch.isfinite(value).all():
                raise FloatingPointError("ADC received nonfinite state; topology was not changed")
        mean_opacity = self.opacity_accum/self.vis_count.clamp_min(1)
        low_opacity = (self.vis_count > 0) & (mean_opacity < self.config.prune_opacity)
        invisible = (step-self.last_seen) >= self.config.invisible_patience
        prune = low_opacity | invisible
        if prune.all():
            raise RuntimeError("ADC would prune all splats; topology was not changed")
        keep = (~prune).nonzero(as_tuple=True)[0]
        score = self.grad_accum/self.vis_count.clamp_min(1)
        candidates = (score > self.config.densify_threshold) & (self.vis_count > 0) & ~prune
        candidates = candidates.nonzero(as_tuple=True)[0] if step <= self.config.densify_until else keep[:0]
        if max_splats is not None:
            room = max(0, max_splats-len(keep))
            if len(candidates) > room:
                order = torch.argsort(score[candidates], descending=True)
                candidates = candidates[order[:room]]
        candidates = candidates.sort().values
        clones, splits = candidates, candidates[:0]
        split_scales = split_rotations = None
        if len(candidates):
            if not torch.isfinite(self.scene_extent) or self.scene_extent <= 0:
                raise ValueError("Clone/split ADC needs a positive fixed training-camera scene extent")
            mask = torch.zeros(len(self.xyz), dtype=torch.bool, device=self.xyz.device)
            mask[candidates] = True
            attrs = self.attributes(camera_center, step, mask)
            if not (torch.isfinite(attrs["scales"]).all() and torch.isfinite(attrs["rotation"]).all()):
                raise FloatingPointError("ADC received nonfinite covariance; topology was not changed")
            large = attrs["scales"].amax(-1) > self.config.adc_split_scale_ratio*self.scene_extent
            clones, splits = candidates[~large], candidates[large]
            split_scales, split_rotations = attrs["scales"][large], attrs["rotation"][large]
        survivor = ~prune
        survivor[splits] = False
        keep = survivor.nonzero(as_tuple=True)[0]
        births = torch.cat([clones, splits.repeat(2)])
        indices = torch.cat([keep, births])
        xyz = self.xyz[indices].clone()
        scales = self.log_scales[indices].clone()
        if len(splits):
            # Sigma = R diag(axis_scales**2) R^T, using physical rendered axes.
            axes = split_scales.repeat(2, 1)
            noise = torch.randn_like(axes)*axes
            first_split = len(keep)+len(clones)
            xyz[first_split:] += torch.bmm(quaternion_matrix(split_rotations).repeat(2, 1, 1),
                                          noise[:, :, None]).squeeze(-1)
            scales[first_split:] -= math.log(1.6)
            if not torch.isfinite(xyz).all():
                raise FloatingPointError("ADC sampled nonfinite coordinates; topology was not changed")
        n_pruned, n_added = int(prune.sum())+len(splits), len(births)
        self.replace_topology(indices, optimizer, xyz, len(keep), log_scales=scales)
        self.last_adc_report = {"cloned": len(clones), "split_parents": len(splits),
                                "split_children": 2*len(splits), "pruned_low_opacity": int(low_opacity.sum()),
                                "pruned_invisible": int(invisible.sum())}
        # Windowed observations reset; survivor timestamps and Adam moments persist.
        self.grad_accum.zero_()
        self.vis_count.zero_()
        self.opacity_accum.zero_()
        if n_added:
            self.last_seen[len(keep):] = step
        self.refresh_bounds()
        return n_added, n_pruned

    @property
    def point_parameter_names(self):
        names = ('xyz', 'features', 'log_scales')
        return names+('rotations',) if self.config.geometry_mode == 'explicit' else names

    @torch.no_grad()
    def freeze_coordinates(self, optimizer=None):
        if bool(self.coordinate_frozen):
            return 0
        if not torch.isfinite(self.xyz).all():
            raise RuntimeError("Coordinate quantization rejected nonfinite positions")
        lo = self.xyz.amin(0)
        hi = torch.maximum(self.xyz.amax(0), lo+1e-6)
        cells, centers = quantize_cells(self.xyz, lo, hi, self.config.octree_depth)
        codes = morton_codes(cells, self.config.octree_depth)
        order = torch.argsort(codes, stable=True)
        sorted_codes = codes[order]
        keep = torch.ones_like(sorted_codes, dtype=torch.bool)
        keep[1:] = sorted_codes[1:] != sorted_codes[:-1]
        indices = order[keep]
        removed = len(self.xyz)-len(indices)
        fraction = removed/len(self.xyz)
        if fraction > self.config.coordinate_max_collision_fraction:
            raise RuntimeError(f"Coordinate quantization would remove {removed}/{len(self.xyz)} "
                               f"splats ({fraction:.2%}), above the collision limit "
                               f"{self.config.coordinate_max_collision_fraction:.2%}; "
                               f"bbox extent={(hi-lo).tolist()}. Inspect geometry before compression.")
        # Commit only after the collision check; a rejection leaves topology,
        # bounds, freeze flags and optimizer state untouched.
        self.bounds_min.copy_(lo)
        self.bounds_max.copy_(hi)
        self.replace_topology(indices, optimizer, centers[indices])
        self.cells = cells[indices]
        self.coordinate_frozen.fill_(True)
        self.xyz.requires_grad_(False)
        return removed
