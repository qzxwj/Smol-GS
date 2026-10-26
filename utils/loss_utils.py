"""3DGS photometric loss and per-image evaluation."""
import math
import torch
import torch.nn.functional as F


def ssim(image, target, window_size=11):
    if image.ndim == 3:
        image, target = image[None], target[None]
    channels = image.shape[1]
    coordinates = torch.arange(window_size, device=image.device, dtype=image.dtype)-(window_size//2)
    gaussian = torch.exp(-coordinates.square()/(2*1.5**2))
    gaussian = gaussian/gaussian.sum()
    window = (gaussian[:, None]*gaussian[None, :])[None, None].expand(channels, 1, -1, -1).contiguous()
    def blur(value):
        return F.conv2d(value, window, padding=window_size//2, groups=channels)
    mean_i, mean_t = blur(image), blur(target)
    var_i = blur(image.square())-mean_i.square()
    var_t = blur(target.square())-mean_t.square()
    cov = blur(image*target)-mean_i*mean_t
    return (((2*mean_i*mean_t+0.01**2)*(2*cov+0.03**2)) /
            ((mean_i.square()+mean_t.square()+0.01**2)*(var_i+var_t+0.03**2))).mean()


def psnr(image, target):
    return -10*torch.log10(F.mse_loss(image, target).clamp_min(1e-12))


def loss_terms(model, package, target, step):
    config = model.config
    image, attrs = package['image'], package['attributes']
    l1 = (image-target).abs().mean()
    dssim = 1-ssim(image, target)
    opacity = attrs['opacity'].sum() if config.stage(step) == 'compaction' else image.new_zeros(())
    rate = attrs['rate'] if step > config.compact_until else image.new_zeros(())
    total = (1-config.lambda_ssim)*l1+config.lambda_ssim*dssim+config.lambda_opacity*opacity+config.lambda_rate*rate
    return total, {'l1': l1, 'dssim': dssim, 'opacity': opacity, 'rate_bits_per_splat': rate}


class LPIPS:
    """Match Inria 3DGS: VGG v0.1 on RGB tensors in [0,1].

    Inria passes image tensors directly to lpipsPyTorch. The installed lpips
    package is numerically equivalent with normalize=False; do not apply 2*x-1.
    Weights must be available locally.
    """
    protocol = 'inria-3dgs-vgg-v0.1-direct-01'
    status = 'VGG v0.1, Inria 3DGS: direct RGB [0,1], normalize=False'

    def __init__(self, device):
        import lpips
        self.network = lpips.LPIPS(net='vgg', version='0.1').to(device).eval()

    @torch.no_grad()
    def __call__(self, image, target):
        return self.network(image[None], target[None], normalize=False).mean()
