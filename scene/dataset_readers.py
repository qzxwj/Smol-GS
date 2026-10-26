"""Read undistorted COLMAP scenes without modifying the dataset."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image
import torch
from .colmap_loader import (read_extrinsics_binary, read_extrinsics_text,
    read_intrinsics_binary, read_intrinsics_text, read_points3D_binary,
    read_points3D_text, qvec2rotmat)


@dataclass
class Camera:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    world_to_camera: torch.Tensor
    image_path: str = ""
    white_background: bool = False

    @property
    def center(self):
        return -self.world_to_camera[:3, :3].T @ self.world_to_camera[:3, 3]

    def to(self, device):
        return Camera(self.name, self.width, self.height, self.fx, self.fy,
                      self.world_to_camera.to(device), self.image_path, self.white_background)

    def image(self, device="cpu"):
        return _load_image(self.image_path, self.width, self.height, self.white_background).to(device)


@lru_cache(maxsize=32)
def _load_image(path, width, height, white):
    with Image.open(path) as image:
        image = image.convert('RGBA').resize((width, height), Image.Resampling.LANCZOS)
        rgba = np.array(image, dtype=np.float32)/255
    rgb = rgba[..., :3]*rgba[..., 3:]+float(white)*(1-rgba[..., 3:])
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1)


def load_points(source):
    source = Path(source)
    sparse = source/'sparse/0'
    if not sparse.exists():
        sparse = source/'sparse'
    for suffix, reader in [('bin', read_points3D_binary), ('txt', read_points3D_text)]:
        path = sparse/f'points3D.{suffix}'
        if path.is_file():
            return reader(str(path))[0].astype(np.float32)
    for path in [sparse/'points3D.ply', source/'points3d.ply']:
        if path.is_file():
            from plyfile import PlyData
            vertices = PlyData.read(str(path))['vertex']
            return np.column_stack([vertices[a] for a in ('x', 'y', 'z')]).astype(np.float32)
    raise FileNotFoundError(f'No SfM point cloud in {source}')


def _size(width, height, resolution):
    if resolution in (1, 2, 4, 8):
        factor = resolution
    elif resolution == -1:
        factor = max(1, width/1600)
    elif resolution > 8:
        factor = width/resolution
    else:
        raise ValueError('resolution must be -1, 1/2/4/8 or target width > 8')
    return max(1, round(width/factor)), max(1, round(height/factor))


def load_scene(source, images='images', resolution=-1, holdout=8, white_background=False):
    source = Path(source).resolve()
    if holdout < 2:
        raise ValueError('holdout must be >=2 so training and test views are disjoint')
    if (source/'transforms_train.json').is_file():
        return _load_blender(source, resolution, white_background)
    sparse = source/'sparse/0'
    if not sparse.is_dir():
        sparse = source/'sparse'
    if (sparse/'images.bin').is_file():
        extrinsics = read_extrinsics_binary(str(sparse/'images.bin'))
        intrinsics = read_intrinsics_binary(str(sparse/'cameras.bin'))
    else:
        extrinsics = read_extrinsics_text(str(sparse/'images.txt'))
        intrinsics = read_intrinsics_text(str(sparse/'cameras.txt'))
    cameras = []
    for extr in sorted(extrinsics.values(), key=lambda camera: camera.name):
        intr = intrinsics[extr.camera_id]
        if intr.model == 'SIMPLE_PINHOLE':
            fx = fy = intr.params[0]
        elif intr.model == 'PINHOLE':
            fx, fy = intr.params[:2]
        else:
            raise ValueError(f'{intr.model} images must be undistorted before training')
        image_path = source/images/extr.name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            original_width, original_height = image.size
        width, height = _size(original_width, original_height, resolution)
        w2c = torch.eye(4)
        w2c[:3, :3] = torch.from_numpy(qvec2rotmat(extr.qvec).astype(np.float32))
        w2c[:3, 3] = torch.tensor(extr.tvec, dtype=torch.float32)
        cameras.append(Camera(extr.name, width, height, float(fx*width/intr.width),
                              float(fy*height/intr.height), w2c, str(image_path), white_background))
    train = [camera for i, camera in enumerate(cameras) if i % holdout]
    test = [camera for i, camera in enumerate(cameras) if i % holdout == 0]
    if not train or not test:
        raise ValueError('Scene must contain nonempty train and test splits')
    return train, test


def _load_blender(source, resolution, white):
    result = []
    for split in ('train', 'test'):
        meta = json.loads((source/f'transforms_{split}.json').read_text())
        cameras = []
        for frame in meta['frames']:
            path = source/frame['file_path']
            if not path.suffix:
                path = path.with_suffix('.png')
            with Image.open(path) as image:
                width, height = _size(*image.size, resolution)
            fx = width/(2*math.tan(meta['camera_angle_x']/2))
            c2w = np.asarray(frame['transform_matrix'], dtype=np.float32)
            c2w[:3, 1:3] *= -1
            w2c = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32))
            cameras.append(Camera(path.stem, width, height, fx, fx, w2c, str(path), white))
        result.append(cameras)
    return tuple(result)
