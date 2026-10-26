"""Neural splats, entropy prior, topology updates and Adam state migration."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.graphics_utils import morton_codes, positional_bits, quantize_cells, quaternion_matrix
from utils.hashgrid import MixedHashEncoder


def mlp(inputs, hidden, outputs, activation=None):
    layers = [nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
              nn.ReLU(), nn.Linear(hidden, outputs)]
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)


class PositionalMLP(nn.Sequential):
    """Three affine layers and ReLUs with an optional fixed affine coordinate shift.

    The shift is foldable into the second bias; it adds no learned parameters.
    Its vector is persisted in Config, so the buffer is nonpersistent.
    """
    def __init__(self, modules, hidden_center=()):
        super().__init__(*modules)
        self.use_hidden_center = bool(len(hidden_center))
        center = torch.as_tensor(hidden_center, dtype=self[2].weight.dtype)
        if not self.use_hidden_center:
            center = self[2].weight.new_zeros(self[2].in_features)
        self.register_buffer("hidden_center", center, persistent=False)

    def hidden_input(self, first_preactivation):
        hidden = self[1](first_preactivation)
        return hidden-self.hidden_center if self.use_hidden_center else hidden

    def forward(self, inputs):
        return self[4](self[3](self[2](self.hidden_input(self[0](inputs)))))


def initialize_octree_mlp(module, seed=1729):
    """ReLU initialization for new runs; preserve every other parameter's RNG."""
    device = next(module.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    for layer in (module[0], module[2]):
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu", generator=generator)
        nn.init.zeros_(layer.bias)
    nn.init.xavier_normal_(module[4].weight, generator=generator)
    nn.init.zeros_(module[4].bias)


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
        self.xyz = nn.Parameter(torch.zeros(count, 3))
        self.features = nn.Parameter(torch.zeros(count, config.feature_dim))
        self.log_scales = nn.Parameter(torch.zeros(count, 3))
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
        input_center = (torch.as_tensor(config.octree_input_mean,dtype=self.xyz.dtype)
                        if config.octree_input_mean else torch.full((3*config.octree_depth,),config.octree_input_center))
        self.register_buffer("position_input_center",input_center,persistent=False)
        self.octree_mlp = PositionalMLP(list(mlp(3*config.octree_depth, config.octree_mlp_width, config.octree_feature_dim).children()),
                                        config.octree_hidden_center)
        if config.octree_activation == "leaky":
            self.octree_mlp[1] = nn.LeakyReLU(0.01)
            self.octree_mlp[3] = nn.LeakyReLU(0.01)
        dim = config.feature_dim+config.octree_feature_dim+4
        self.opacity_mlp = mlp(dim, config.mlp_width, 1, nn.Tanh())
        self.color_mlp = mlp(dim, config.mlp_width, 3, nn.Sigmoid())
        self.rotation_mlp = mlp(dim, config.mlp_width, 4)
        self.scale_mlp = mlp(dim, config.mlp_width, 3)
        self.hash_encoder = MixedHashEncoder(config)
        # Table A.2: 2*n_f + (3 mean_s + 3 sigma_s + 1 delta_f + 1 delta_s).
        self.entropy_mlp = mlp(self.hash_encoder.output_dim, config.mlp_width, 2*config.feature_dim+8)
        self.to(device)
        if config.octree_init == "kaiming":
            initialize_octree_mlp(self.octree_mlp, config.octree_init_seed)
        if config.octree_init_gain != 1:
            # ReLU positive homogeneity: preserve the complete initial function
            # while changing the scale of hidden parameters, not the LR table.
            gain = config.octree_init_gain
            with torch.no_grad():
                self.octree_mlp[0].weight.mul_(gain)
                self.octree_mlp[0].bias.mul_(gain)
                self.octree_mlp[2].weight.mul_(gain)
                self.octree_mlp[2].bias.mul_(gain*gain)
                self.octree_mlp[4].weight.div_(gain*gain)
        if config.octree_input_center or config.octree_input_mean:
            # Equivalent initial affine map: W*(bits-c)+(b+W@c) == W*bits+b.
            # Only the optimizer's parameter coordinates change; ReLU and model
            # capacity are retained. Experimental, not an author-confirmed choice.
            with torch.no_grad():
                self.octree_mlp[0].bias.add_(self.octree_mlp[0].weight@self.position_input_center)
        if self.octree_mlp.use_hidden_center:
            with torch.no_grad():
                self.octree_mlp[2].bias.add_(self.octree_mlp[2].weight@self.octree_mlp.hidden_center)

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
        self.reset_statistics(len(points))
        self.cells = torch.zeros_like(points, dtype=torch.long)
        self.refresh_bounds()
        self.hash_min.copy_(self.bounds_min)
        self.hash_max.copy_(self.bounds_max)
        if self.config.octree_input_center_mode == "sfm" and not self.config.octree_input_mean:
            cells, _ = quantize_cells(self.xyz,self.bounds_min,self.bounds_max,self.config.octree_depth)
            center = positional_bits(cells,self.config.octree_depth).mean(0)
            self.octree_mlp[0].bias.add_(self.octree_mlp[0].weight@(center-self.position_input_center))
            self.position_input_center.copy_(center)
            self.config.octree_input_mean = tuple(center.cpu().tolist())
        if self.config.octree_hidden_center_mode == "sfm" and not self.octree_mlp.use_hidden_center:
            cells, _ = quantize_cells(self.xyz, self.bounds_min, self.bounds_max, self.config.octree_depth)
            bits = self.position_inputs(cells)
            center = self.octree_mlp[1](self.octree_mlp[0](bits)).mean(0)
            self.octree_mlp.hidden_center.copy_(center)
            self.octree_mlp.use_hidden_center = True
            self.config.octree_hidden_center = tuple(center.cpu().tolist())
            # Compensation preserves the complete initial function, including
            # both ReLUs, instead of changing random weights or their seed.
            self.octree_mlp[2].bias.add_(self.octree_mlp[2].weight@center)

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
        return {"bbox_extent": (self.xyz.amax(0)-self.xyz.amin(0)).tolist(),
                "controller_scale_max": float(self.log_scales.amax().exp()),
                "scene_extent": float(self.scene_extent)}

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
            first = self.octree_mlp[0](self.position_inputs(cells))
            second = self.octree_mlp[2](self.octree_mlp.hidden_input(first))
            hidden = self.octree_mlp[3](second)
            embedding = self.octree_mlp[4](hidden)
            nonzero_points += int((hidden != 0).any(-1).sum())
            lo, hi = embedding.amin(0), embedding.amax(0)
            minima = lo if minima is None else torch.minimum(minima, lo)
            maxima = hi if maxima is None else torch.maximum(maxima, hi)
            first_positive += int((first > 0).sum())
            second_positive += int((second > 0).sum())
            second_max = max(second_max, float(second.max()))
        total = count*self.config.octree_mlp_width
        return {"octree_probe_points": count, "octree_probe_all_points": count == len(self.xyz),
                "octree_first_active_fraction": first_positive/total,
                "octree_second_active_fraction": second_positive/total,
                "octree_second_max": second_max,
                "octree_hidden_nonzero_point_fraction": nonzero_points/count,
                "octree_embedding_range": float((maxima-minima).amax()),
                "octree_hidden_gradient_norm": sum(float(p.grad.norm()) for layer in
                    (self.octree_mlp[0], self.octree_mlp[2]) for p in layer.parameters() if p.grad is not None)}

    @torch.no_grad()
    def refresh_bounds(self):
        lo, hi = self.xyz.amin(0), self.xyz.amax(0)
        self.bounds_min.copy_(lo)
        self.bounds_max.copy_(torch.maximum(hi, lo+1e-6))

    def prior(self, xyz):
        normalized = (xyz-self.hash_min)/(self.hash_max-self.hash_min).clamp_min(1e-6)
        raw = self.entropy_mlp(self.hash_encoder(normalized))
        d = self.config.feature_dim
        mf, sf, ms, ss, qf, qs = raw.split([d, d, 3, 3, 1, 1], dim=-1)
        # Positive Gaussian std and HAC's bounded step adjustment.
        return (mf, F.softplus(sf)+1e-6, ms, F.softplus(ss)+1e-6,
                self.config.feature_step*(1+torch.tanh(qf)).clamp_min(1e-4),
                self.config.scale_step*(1+torch.tanh(qs)).clamp_min(1e-4))

    def position_inputs(self, cells):
        return positional_bits(cells,self.config.octree_depth,self.xyz.dtype)-self.position_input_center

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
            mf, sf, ms, ss, qf, qs = self.prior(xyz)
            features = quantize(features, qf, self.training)
            scales = quantize(scales, qs, self.training)
            if self.training and len(xyz):
                rate = (gaussian_bits(features, mf, sf, qf, self.config.rate_bin_half_width).sum(-1)
                        + gaussian_bits(scales, ms, ss, qs, self.config.rate_bin_half_width).sum(-1)).mean()
        if bool(self.coordinate_frozen):
            cells = self.cells[mask]
        else:
            cells, _ = quantize_cells(xyz, self.bounds_min, self.bounds_max, self.config.octree_depth)
        embedding = self.position_embedding(cells)
        view = camera_center[None]-xyz
        distance = view.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        augmented = torch.cat([features, view/distance, distance, embedding], dim=-1)
        opacity = self.opacity_mlp(augmented).abs()
        color = self.color_mlp(augmented)
        rotation = F.normalize(self.rotation_mlp(augmented), dim=-1)
        # Noise may cross zero; physical Gaussian scales must stay positive.
        scales = scales.clamp_min(1e-7)*torch.sigmoid(self.scale_mlp(augmented))
        return {"xyz": xyz, "color": color, "opacity": opacity,
                "scales": scales, "rotation": rotation, "rate": rate, "mask": mask}

    def make_optimizer(self):
        groups = []
        for name, (initial, _) in self.config.learning_rates.items():
            value = getattr(self, name)
            params = list(value.parameters()) if isinstance(value, nn.Module) else [value]
            groups.append({"params": params, "lr": initial, "name": name})
        return torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    def update_learning_rates(self, optimizer, step):
        t = min(max(step/self.config.lr_max_steps, 0), 1)
        for group in optimizer.param_groups:
            a, b = self.config.learning_rates[group["name"]]
            group["lr"] = math.exp((1-t)*math.log(a)+t*math.log(b))
            if group["name"] == "xyz" and bool(self.coordinate_frozen):
                group["lr"] = 0.0

    @torch.no_grad()
    def replace_topology(self, indices, optimizer=None, xyz=None, new_start=None, log_scales=None):
        """Reindex splats; carry Adam moments for survivors, zero them for clones."""
        for name in ("xyz", "features", "log_scales"):
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
    def density_control(self, optimizer, camera_center, step, max_splats=None, physical_size=None):
        if not torch.isfinite(self.scene_extent) or self.scene_extent <= 0:
            raise RuntimeError("ADC requires a fixed training-camera scene extent")
        if physical_size is not None and (physical_size.shape != (len(self.xyz),)
                                          or physical_size.device != self.xyz.device):
            raise ValueError("physical_size must be one axis-size maximum per splat on the model device")
        controller_size = self.log_scales.amax(dim=-1).exp()
        finite = torch.isfinite(self.xyz).all(-1) & torch.isfinite(self.log_scales).all(-1)
        finite &= torch.isfinite(controller_size) & torch.isfinite(self.features).all(-1)
        if physical_size is not None:
            finite &= torch.isfinite(physical_size) & (physical_size >= 0)
        # Optional diagnostic input: evaluate an effective-size criterion without
        # changing the default controller criterion or any other ADC behavior.
        size = controller_size if physical_size is None else physical_size
        oversized = (size > self.config.adc_prune_scale_ratio*self.scene_extent
                     if self.config.adc_size_prune_from <= step <= self.config.adc_size_prune_until
                     else torch.zeros_like(finite))
        mean_opacity = self.opacity_accum/self.vis_count.clamp_min(1)
        prune = ((self.vis_count > 0) & (mean_opacity < self.config.prune_opacity))
        prune |= (step-self.last_seen) >= self.config.invisible_patience
        prune |= ~finite | oversized
        # Keep one valid splat for opacity/invisibility pruning; never rescue a
        # nonfinite or oversized point just to keep the renderer running.
        if prune.all():
            valid = finite & ~oversized
            if not valid.any():
                raise RuntimeError("ADC rejected all splats as nonfinite or oversized")
            prune[mean_opacity.masked_fill(~valid, -float('inf')).argmax()] = False
        keep = (~prune).nonzero(as_tuple=True)[0]
        candidates = (self.grad_accum/self.vis_count.clamp_min(1) >= self.config.densify_threshold) & ~prune
        candidates = candidates.nonzero(as_tuple=True)[0] if step <= self.config.densify_until else keep[:0]
        if max_splats is not None:
            room = max(0, max_splats-len(keep))
            if len(candidates) > room:
                order = torch.argsort(self.grad_accum[candidates]/self.vis_count[candidates].clamp_min(1), descending=True)
                candidates = candidates[order[:room]]
        candidates = candidates.sort().values
        # Inria 3DGS: clone small splats, replace a large parent with two
        # children whose scales shrink by 0.8*N. The controller is a conservative
        # view-independent upper bound on Smol-GS's predicted covariance axes.
        split_mask = controller_size[candidates] > self.config.adc_split_scale_ratio*self.scene_extent
        clones, splits = candidates[~split_mask], candidates[split_mask]
        survivor_mask = ~prune
        survivor_mask[splits] = False
        keep = survivor_mask.nonzero(as_tuple=True)[0]
        births = torch.cat([clones, splits.repeat(2)])
        indices = torch.cat([keep, births])
        xyz = self.xyz[indices].clone()
        scales = self.log_scales[indices].clone()
        if len(splits):
            mask = torch.zeros(len(self.xyz), dtype=torch.bool, device=self.xyz.device)
            mask[splits] = True
            attrs = self.attributes(camera_center, step, mask)
            noise = torch.randn((2*len(splits), 3), device=self.xyz.device)*attrs["scales"].repeat(2, 1)
            first_child = len(keep)+len(clones)
            xyz[first_child:] += torch.bmm(quaternion_matrix(attrs["rotation"]).repeat(2, 1, 1),
                                          noise[:, :, None]).squeeze(-1)
            scales[first_child:] -= math.log(1.6)
        n_pruned, n_added = int(prune.sum())+len(splits), len(births)
        self.replace_topology(indices, optimizer, xyz, len(keep), log_scales=scales)
        self.last_adc_report = {"cloned": len(clones), "split_parents": len(splits),
                                "pruned_oversized": int(oversized.sum()),
                                "pruned_nonfinite": int((~finite).sum())}
        # Opacity/gradient observations are windowed; unseen timestamps persist.
        self.grad_accum.zero_()
        self.vis_count.zero_()
        self.opacity_accum.zero_()
        if n_added:
            self.last_seen[len(keep):] = step
        self.refresh_bounds()
        return n_added, n_pruned

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
