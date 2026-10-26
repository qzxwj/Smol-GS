"""Hyperparameters transcribed from arxiv.tex, sections 3 and A."""
from dataclasses import asdict, dataclass, field
import json
import math


@dataclass
class Config:
    feature_dim: int = 8
    octree_depth: int = 16
    octree_feature_dim: int = 8
    mlp_width: int = 128
    octree_mlp_width: int = 64
    hash_features: int = 4
    hash_log2_3d: int = 13
    hash_log2_2d: int = 15
    resolutions_3d: tuple = (18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514)
    resolutions_2d: tuple = (130, 258, 514, 1026)
    iterations: int = 35000
    warmup_until: int = 500
    densify_until: int = 15000
    compact_until: int = 20000
    feature_until: int = 30000
    lr_max_steps: int = 30000
    lambda_ssim: float = 0.2
    lambda_opacity: float = 3e-7
    lambda_rate: float = 5e-4
    densify_threshold: float = 2e-4
    prune_opacity: float = 0.005
    # The following choices are not specified in the paper; see docs/FIDELITY.md.
    adc_interval: int = 100
    invisible_patience: int = 1000
    adc_split_scale_ratio: float = 0.01
    adc_prune_scale_ratio: float = 0.1
    adc_size_prune_from: int = 3000
    adc_size_prune_until: int = 20000
    octree_init: str = "torch"
    octree_init_seed: int = 1729
    octree_init_gain: float = 1.0
    octree_activation: str = "relu"
    feature_init_std: float = 0.01
    octree_input_center: float = 0.0
    octree_input_center_mode: str = "constant"
    octree_input_mean: tuple = ()
    octree_hidden_center_mode: str = "none"
    octree_hidden_center: tuple = ()
    coordinate_max_collision_fraction: float = 0.5
    feature_step: float = 1.0
    scale_step: float = 0.001
    rate_bin_half_width: float = 0.5
    hash_backend: str = "auto"
    learning_rates: dict = field(default_factory=lambda: {
        "xyz": [0.0002, 1e-5], "features": [0.0075, 0.0075],
        "log_scales": [0.01, 0.002], "opacity_mlp": [0.002, 2e-5],
        "color_mlp": [0.008, 5e-5], "rotation_mlp": [0.004, 0.004],
        "scale_mlp": [0.004, 0.004], "entropy_mlp": [0.005, 1e-5],
        "octree_mlp": [0.008, 0.008], "hash_encoder": [0.005, 1e-5],
    })

    def __post_init__(self):
        if not 1 <= self.octree_depth <= 20:
            raise ValueError("octree_depth must be in [1, 20] (Morton codes use int64)")
        if not 0 <= self.warmup_until < self.densify_until < self.compact_until < self.feature_until < self.iterations:
            raise ValueError("The five training stages must be strictly ordered")
        if self.hash_backend not in ("auto", "cuda", "torch"):
            raise ValueError("hash_backend must be auto, cuda or torch")
        if self.adc_interval < 1 or self.invisible_patience < 1:
            raise ValueError("ADC interval and invisible patience must be positive")
        if not 0 < self.adc_split_scale_ratio < self.adc_prune_scale_ratio:
            raise ValueError("ADC scale ratios must satisfy 0 < split < prune")
        if not math.isfinite(self.octree_init_gain) or self.octree_init_gain <= 0:
            raise ValueError("octree_init_gain must be finite and positive")
        if self.octree_input_center_mode not in ("constant", "sfm", "checkpoint"):
            raise ValueError("octree_input_center_mode must be constant, sfm or checkpoint")
        if self.octree_input_center_mode == "checkpoint" and not self.octree_input_mean:
            raise ValueError("Checkpoint bit centering requires the recorded vector")
        if self.octree_input_mean and (self.octree_input_center_mode not in ("sfm", "checkpoint") or
                len(self.octree_input_mean) != 3*self.octree_depth or
                not all(math.isfinite(float(x)) and 0 <= x <= 1 for x in self.octree_input_mean)):
            raise ValueError("Invalid fixed bit centering vector")
        if self.octree_hidden_center_mode not in ("none", "sfm", "checkpoint"):
            raise ValueError("octree_hidden_center_mode must be none, sfm or checkpoint")
        if self.octree_hidden_center_mode == "checkpoint" and not self.octree_hidden_center:
            raise ValueError("Checkpoint hidden centering requires the recorded vector")
        if self.octree_hidden_center and (self.octree_hidden_center_mode not in ("sfm", "checkpoint") or
                len(self.octree_hidden_center) != self.octree_mlp_width or
                not all(math.isfinite(float(x)) for x in self.octree_hidden_center)):
            raise ValueError("Invalid fixed hidden centering vector")
        if not 0 <= self.octree_input_center <= 1:
            raise ValueError("octree_input_center must be in [0, 1]")
        if self.octree_activation not in ("relu", "leaky") or self.feature_init_std <= 0:
            raise ValueError("Invalid positional activation or feature initialization std")
        if self.octree_init not in ("torch", "kaiming"):
            raise ValueError("octree_init must be torch or kaiming")
        if self.adc_size_prune_until < self.adc_size_prune_from:
            raise ValueError("Size-prune end must not precede its start")
        if self.adc_size_prune_from < 0 or not 0 <= self.coordinate_max_collision_fraction < 1:
            raise ValueError("Invalid size-prune start or coordinate collision limit")
        if self.feature_step <= 0 or self.scale_step <= 0 or self.rate_bin_half_width <= 0:
            raise ValueError("Quantization steps and rate bin half width must be positive")

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
    def load(cls, path=None, variant="base"):
        values = json.loads(open(path).read()) if path else {}
        if not path:
            values["lambda_opacity"] = {"small": 4e-7, "base": 3e-7, "large": 1e-7}[variant]
        return cls(**values)


def add_model_optimization_arguments(parser):
    """Accept 3DGS/HAC-style underscore flags as well as hyphenated flags."""
    group = parser.add_argument_group('Smol-GS model and optimization overrides')
    aliases = {
        'feature_dim': ['--feat_dim'], 'octree_depth': ['--recursion'],
        'lambda_ssim': ['--lambda_dssim'], 'lambda_opacity': ['--lambda_o'],
        'lambda_rate': ['--lambda_q', '--lmbda'], 'adc_interval': ['--update_interval'],
        'densify_threshold': ['--densify_grad_threshold'], 'prune_opacity': ['--min_opacity'],
        'densify_until': ['--densify_until_iter'], 'warmup_until': ['--densify_from_iter'],
    }
    default = Config()
    for name, value in default.to_dict().items():
        if name in ('learning_rates', 'resolutions_2d', 'resolutions_3d', 'octree_hidden_center', 'octree_input_mean'):
            continue
        options = list(dict.fromkeys(['--'+name.replace('_', '-'), '--'+name]+aliases.get(name, [])))
        group.add_argument(*options, dest='config_'+name, type=type(value), default=None)
    conventional = {'xyz': 'position', 'features': 'feature', 'log_scales': 'scaling',
                    'opacity_mlp': 'mlp_opacity', 'color_mlp': 'mlp_color',
                    'rotation_mlp': 'mlp_rotation', 'scale_mlp': 'mlp_scaling',
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
