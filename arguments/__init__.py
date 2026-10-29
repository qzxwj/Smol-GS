"""Hyperparameters transcribed from arxiv.tex, sections 3 and A."""
from dataclasses import asdict, dataclass, field
from argparse import BooleanOptionalAction
from pathlib import Path
import json
import math


@dataclass
class Config:
    feature_dim: int = 8
    octree_depth: int = 16
    octree_feature_dim: int = 8
    mlp_width: int = 128
    octree_mlp_width: int = 64
    appearance_backend: str = "torch"
    opacity_activation: str = "abs_tanh"
    geometry_mode: str = "explicit"
    explicit_scaling_lr: float = 0.005
    explicit_rotation_lr: float = 0.001
    geometry_quantization: str = 'none'
    vq_start_iter: int = 20001
    rate_constrain_iter: int = 25001
    vq_scale_cbsize: int = 8192
    vq_rot_cbsize: int = 8192
    vq_scale_lmbda: float = 4096.
    vq_rot_lmbda: float = 256.
    vq_cb_lr: float = 0.0002
    vq_logits_lr: float = 0.002
    vq_patch_size: int = 16384
    reactivate_codeword_period: int = 1000
    iterations: int = 35000
    warmup_until: int = 500
    densify_until: int = 15000
    compact_until: int = 20000
    feature_until: int = 30000
    lr_max_steps: int = 30000
    lambda_ssim: float = 0.2
    lambda_rate: float = 5e-4
    densify_threshold: float = 2e-4
    prune_opacity: float = 0.005
    # The following choices are not specified in the paper; see docs/FIDELITY.md.
    adc_interval: int = 100
    invisible_patience: int = 1000
    # Inria 3DGS size boundary, applied to current-camera rendered axes.
    adc_split_scale_ratio: float = 0.01
    octree_backend: str = "torch"
    octree_init_seed: int = 1337
    feature_init_std: float = 0.01
    coordinate_max_collision_fraction: float = 0.5
    # Coordinate coding is always independent of the positional MLP.
    coordinate_depth: int = 16
    coordinate_tail_fraction: float = 0.0
    feature_step: float = 1.0
    rate_bin_half_width: float = 0.5
    hash_backend: str = "auto"
    hash_detach_coordinates: bool = True
    hash_features: int = 4
    hash_log2_3d: int = 13
    hash_log2_2d: int = 15
    resolutions_3d: tuple = (18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514)
    resolutions_2d: tuple = (130, 258, 514, 1026)
    learning_rates: dict = field(default_factory=lambda: {
        "xyz": [0.0002, 1e-5], "features": [0.0075, 0.0075],
        "opacity_mlp": [0.002, 2e-5],
        "color_mlp": [0.008, 5e-5], "entropy_mlp": [0.005, 1e-5],
        "octree_mlp": [0.008, 0.008], "hash_encoder": [0.005, 1e-5],
    })

    def __post_init__(self):
        if self.geometry_mode != 'explicit':
            raise ValueError('Only explicit geometry is supported')
        unknown_rates = set(self.learning_rates)-{'xyz', 'features', 'opacity_mlp',
                                                  'color_mlp', 'entropy_mlp', 'octree_mlp', 'hash_encoder'}
        if unknown_rates:
            raise ValueError(f'Unknown learning-rate groups: {sorted(unknown_rates)}')
        if self.geometry_quantization not in ('none', 'ecvq'):
            raise ValueError('geometry_quantization must be none or ecvq')
        if self.geometry_quantization == 'ecvq':
            if not self.compact_until < self.vq_start_iter <= self.rate_constrain_iter <= self.iterations:
                raise ValueError('ECVQ must start after compaction; start <= rate_constrain <= iterations')
            if any(not isinstance(value, int) or isinstance(value, bool)
                   for value in (self.vq_start_iter, self.rate_constrain_iter)):
                raise ValueError('ECVQ stage iterations must be integers')
        for name in ('vq_scale_cbsize', 'vq_rot_cbsize', 'vq_patch_size', 'reactivate_codeword_period'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('vq_scale_lmbda', 'vq_rot_lmbda', 'vq_cb_lr', 'vq_logits_lr'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('explicit_scaling_lr', 'explicit_rotation_lr'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.opacity_activation != 'abs_tanh':
            raise ValueError('Only paper opacity_activation=abs_tanh is supported')
        if not 1 <= self.octree_depth <= 20:
            raise ValueError("octree_depth must be in [1, 20] (Morton codes use int64)")
        if not isinstance(self.coordinate_depth, int) or not 1 <= self.coordinate_depth <= 20:
            raise ValueError('coordinate_depth must be in [1, 20]')
        if not 0 <= self.coordinate_tail_fraction < .5:
            raise ValueError('coordinate_tail_fraction must be in [0, .5)')
        if not 0 <= self.warmup_until < self.densify_until < self.compact_until < self.feature_until < self.iterations:
            raise ValueError("The five training stages must be strictly ordered")
        if self.hash_backend not in ("auto", "cuda", "torch"):
            raise ValueError("Binary hash_backend must be auto, cuda or torch")
        if not self.hash_detach_coordinates:
            raise ValueError('Binary hash requires hash_detach_coordinates=True')
        if (self.hash_features not in (1, 2, 4, 8) or
                not 1 <= self.hash_log2_3d <= 24 or not 1 <= self.hash_log2_2d <= 24):
            raise ValueError('Invalid binary hash table configuration')
        for name in ('resolutions_3d', 'resolutions_2d'):
            values = tuple(getattr(self, name))
            if not values or any(not isinstance(r, int) or r < 3 for r in values):
                raise ValueError('Hash resolutions must be integers >= 3')
            setattr(self, name, values)
        if self.adc_interval < 1 or self.invisible_patience < 1:
            raise ValueError("ADC interval and invisible patience must be positive")
        if not math.isfinite(self.adc_split_scale_ratio) or self.adc_split_scale_ratio <= 0:
            raise ValueError("ADC split scale ratio must be finite and positive")
        if not math.isfinite(self.feature_init_std) or self.feature_init_std <= 0:
            raise ValueError("feature_init_std must be finite and positive")
        if self.octree_backend not in ('torch', 'tcnn'):
            raise ValueError('octree_backend must be torch or tcnn')
        if self.octree_backend == 'tcnn':
            if self.octree_mlp_width not in (16, 32, 64, 128):
                raise ValueError('FullyFusedMLP width must be 16, 32, 64 or 128')
        if self.appearance_backend not in ('torch', 'tcnn'):
            raise ValueError('appearance_backend must be torch or tcnn')
        if self.appearance_backend == 'tcnn' and self.mlp_width not in (16, 32, 64, 128):
            raise ValueError('FullyFusedMLP width must be 16, 32, 64 or 128')
        if not 0 <= self.coordinate_max_collision_fraction < 1:
            raise ValueError("Invalid coordinate collision limit")
        if self.feature_step <= 0 or self.rate_bin_half_width <= 0:
            raise ValueError("Quantization steps and rate bin half width must be positive")

    def optimizer_rates(self):
        """Keep appearance schedules; explicit geometry uses RDO's fixed rates."""
        rates = {name: list(pair) for name, pair in self.learning_rates.items()}
        rates['log_scales'] = [self.explicit_scaling_lr]*2
        rates['rotations'] = [self.explicit_rotation_lr]*2
        return rates

    def stage(self, step):
        if step <= self.warmup_until:
            return "warmup"
        if step <= self.densify_until:
            return "densification"
        if step <= self.compact_until:
            return "compaction"
        if step <= self.feature_until:
            return "feature_compression"
        return "coordinate_compression"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_saved(cls, values):
        if values.get('opacity_activation') != 'abs_tanh':
            raise ValueError('Saved model must record paper opacity_activation=abs_tanh')
        values = dict(values)
        return cls(**values)

    @classmethod
    def load(cls, path=None, variant="base"):
        values = json.loads(Path(path).read_text()) if path else {}
        values = dict(values)
        return cls(**values)


def add_model_optimization_arguments(parser):
    """Accept 3DGS/HAC-style underscore flags as well as hyphenated flags."""
    group = parser.add_argument_group('Smol-GS model and optimization overrides')
    aliases = {
        'feature_dim': ['--feat_dim'], 'octree_depth': ['--recursion'],
        'lambda_ssim': ['--lambda_dssim'],
        'lambda_rate': ['--lambda_q', '--lmbda'], 'adc_interval': ['--update_interval'],
        'densify_threshold': ['--densify_grad_threshold'], 'prune_opacity': ['--min_opacity'],
        'densify_until': ['--densify_until_iter'], 'warmup_until': ['--densify_from_iter'],
    }
    default = Config()
    for name, value in default.to_dict().items():
        if name in ('learning_rates', 'resolutions_3d', 'resolutions_2d'):
            continue
        options = list(dict.fromkeys(['--'+name.replace('_', '-'), '--'+name]+aliases.get(name, [])))
        if isinstance(value, bool):
            group.add_argument(*options, dest='config_'+name, action=BooleanOptionalAction, default=None)
        else:
            group.add_argument(*options, dest='config_'+name, type=type(value), default=None)
    conventional = {'xyz': 'position', 'features': 'feature',
                    'opacity_mlp': 'mlp_opacity', 'color_mlp': 'mlp_color',
                    'entropy_mlp': 'mlp_grid', 'octree_mlp': 'mlp_octree',
                    'hash_encoder': 'encoding_xyz'}
    for parameter in default.learning_rates:
        for index, endpoint in enumerate(('init', 'final')):
            group.add_argument(f'--{parameter}_lr_{endpoint}', f'--{conventional[parameter]}_lr_{endpoint}',
                               dest=f'lr_{parameter}_{index}',
                               type=float, default=None)


def config_from_args(args):
    config = Config.load(args.config, args.variant)
    values = config.to_dict()
    for name in list(values):
        override = getattr(args, 'config_'+name, None)
        if override is not None:
            values[name] = override
    for parameter, pair in values['learning_rates'].items():
        for index in (0, 1):
            override = getattr(args, f'lr_{parameter}_{index}', None)
            if override is not None:
                pair[index] = override
    return Config(**values)
