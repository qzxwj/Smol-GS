"""Original Inria/HAC CUDA rasterizer plus a small differentiable CPU reference."""
import torch
from utils.graphics_utils import quaternion_matrix


def projection(camera):
    znear, zfar = 0.01, 100.0
    p = camera.world_to_camera.new_zeros(4, 4)
    p[0, 0] = 2*camera.fx/camera.width
    p[1, 1] = 2*camera.fy/camera.height
    p[2, 2] = zfar/(zfar-znear)
    p[2, 3] = -zfar*znear/(zfar-znear)
    p[3, 2] = 1
    return p


def render(model, camera, step=0, backend='cuda', background=None):
    attrs = model.attributes(camera.center, step)
    if background is None:
        background = model.xyz.new_full((3,), float(camera.white_background))
    if backend == 'cuda':
        if not model.xyz.is_cuda:
            raise ValueError('CUDA rendering requires a CUDA model')
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        view = camera.world_to_camera.T.contiguous()
        full_projection = view @ projection(camera).T
        settings = GaussianRasterizationSettings(
            image_height=camera.height, image_width=camera.width,
            tanfovx=camera.width/(2*camera.fx), tanfovy=camera.height/(2*camera.fy),
            bg=background, scale_modifier=1.0, viewmatrix=view,
            projmatrix=full_projection, sh_degree=0, campos=camera.center,
            prefiltered=False, debug=False)
        screen = torch.zeros_like(attrs['xyz'], requires_grad=torch.is_grad_enabled())
        if screen.requires_grad:
            screen.retain_grad()
        image, radii = GaussianRasterizer(settings)(
            means3D=attrs['xyz'], means2D=screen, shs=None,
            colors_precomp=attrs['color'], opacities=attrs['opacity'],
            scales=attrs['scales'], rotations=attrs['rotation'], cov3D_precomp=None)
        return {'image': image, 'visible': radii > 0, 'radii': radii, 'attributes': attrs,
                'viewspace_points': screen}
    if backend != 'torch':
        raise ValueError('renderer must be cuda or torch')
    if len(model.xyz)*camera.width*camera.height > 8_000_000:
        raise RuntimeError('The torch reference renderer is for tiny validation scenes only; use CUDA for benchmarks')
    return _reference_render(attrs, camera, background)


def _reference_render(attrs, camera, background):
    xyz = attrs['xyz']
    w2c = camera.world_to_camera
    camera_xyz = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    x, y, z = camera_xyz.unbind(-1)
    z_safe = z.clamp_min(1e-4)
    # Inria's NDC->pixel convention yields center (W-1)/2, (H-1)/2.
    means = torch.stack([camera.fx*x/z_safe+(camera.width-1)/2,
                         camera.fy*y/z_safe+(camera.height-1)/2], dim=-1)
    # CUDA exposes projected-position derivatives in its NDC convention.
    # A zero-valued leaf keeps the reference image unchanged and supplies the
    # same W/2,H/2 gradient scaling as the CUDA backward kernel.
    screen = torch.zeros_like(xyz, requires_grad=torch.is_grad_enabled())
    if screen.requires_grad:
        screen.retain_grad()
    means = means+screen[:, :2]*xyz.new_tensor([camera.width/2, camera.height/2])
    rotations = quaternion_matrix(attrs['rotation'])
    factors = rotations*attrs['scales'][:, None, :]
    covariance = factors @ factors.transpose(1, 2)
    covariance = w2c[:3, :3] @ covariance @ w2c[:3, :3].T
    # Inria clamps the projection Jacobian to 1.3*tan(FOV/2).
    rx = (x/z_safe).clamp(-1.3*camera.width/(2*camera.fx), 1.3*camera.width/(2*camera.fx))
    ry = (y/z_safe).clamp(-1.3*camera.height/(2*camera.fy), 1.3*camera.height/(2*camera.fy))
    zero = torch.zeros_like(z)
    jacobian = torch.stack([camera.fx/z_safe, zero, -camera.fx*rx/z_safe,
                            zero, camera.fy/z_safe, -camera.fy*ry/z_safe], dim=-1).reshape(-1, 2, 3)
    covariance_2d = jacobian @ covariance @ jacobian.transpose(1, 2)
    covariance_2d = covariance_2d+torch.eye(2, device=xyz.device)*0.3
    inverse = torch.linalg.inv(covariance_2d)
    radii = (torch.linalg.eigvalsh(covariance_2d)[:, -1].sqrt()*3).ceil()
    visible = ((z > 0.2) & (means[:, 0]+radii >= 0) & (means[:, 0]-radii < camera.width)
               & (means[:, 1]+radii >= 0) & (means[:, 1]-radii < camera.height))
    radii = radii*visible
    yy, xx = torch.meshgrid(torch.arange(camera.height, device=xyz.device),
                            torch.arange(camera.width, device=xyz.device), indexing='ij')
    pixels = torch.stack([xx, yy], dim=-1).to(xyz.dtype)
    delta = pixels[None]-means[:, None, None]
    power = -0.5*torch.einsum('nhwi,nij,nhwj->nhw', delta, inverse, delta)
    alpha = (attrs['opacity'][:, :, None]*power.exp()).clamp(max=0.99)
    alpha = torch.where((alpha >= 1/255) & visible[:, None, None], alpha, 0)
    order = torch.argsort(z)
    alpha = alpha[order]
    transmission = torch.cumprod(torch.cat([torch.ones_like(alpha[:1]), 1-alpha], dim=0), dim=0)
    weights = transmission[:-1]*alpha
    image = torch.einsum('nhw,nc->chw', weights, attrs['color'][order])+transmission[-1]*background[:, None, None]
    return {'image': image, 'visible': visible, 'radii': radii, 'attributes': attrs,
            'viewspace_points': screen}
