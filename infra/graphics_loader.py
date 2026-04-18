"""Graphics scene/model loaders for 3D Gaussian Splatting."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from gsplat.rendering import rasterization
from huggingface_hub import snapshot_download


_CAM_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class GaussianScene:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    viewmats: torch.Tensor
    Ks: torch.Tensor
    width: int
    height: int
    scene_name: str


def is_3dgs_model(model_name: str) -> bool:
    name = model_name.lower()
    return "3dgs" in name or "gaussian" in name or "splat" in name


def _download_scene(scene_name: str) -> Path:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    path = snapshot_download(
        "nerfstudioteam/datasets",
        repo_type="dataset",
        allow_patterns=[
            f"{scene_name}/base_cam.json",
            f"{scene_name}/colmap/sparse/0/*",
        ],
        max_workers=1,
    )
    return Path(path) / scene_name / "colmap" / "sparse" / "0"


def _qvec_to_rotmat(qvec: tuple[float, float, float, float]) -> torch.Tensor:
    w, x, y, z = qvec
    return torch.tensor(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=torch.float32,
    )


def _read_camera(path: Path) -> tuple[int, int, torch.Tensor]:
    with open(path, "rb") as f:
        _num_cameras = struct.unpack("<Q", f.read(8))[0]
        _cam_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
        _, nparams = _CAM_MODELS[model_id]
        params = struct.unpack("<" + "d" * nparams, f.read(8 * nparams))
    fx, fy, cx, cy = params[:4]
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return width, height, K


def _read_images(path: Path, limit: int | None = None) -> list[torch.Tensor]:
    viewmats = []
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for idx in range(num_images):
            _image_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<dddd", f.read(32))
            tvec = struct.unpack("<ddd", f.read(24))
            _camera_id = struct.unpack("<i", f.read(4))[0]
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes.extend(c)
            _name = name_bytes.decode("utf-8")
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(num_points2d * 24, os.SEEK_CUR)

            view = torch.eye(4, dtype=torch.float32)
            view[:3, :3] = _qvec_to_rotmat(qvec)
            view[:3, 3] = torch.tensor(tvec, dtype=torch.float32)
            viewmats.append(view)
            if limit is not None and len(viewmats) >= limit:
                break
    return viewmats


def _read_points(path: Path, max_points: int | None = None) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    xyzs = []
    rgbs = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            _point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("<BBB", f.read(3))
            _error = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.seek(track_len * 8, os.SEEK_CUR)
            xyzs.append(xyz)
            rgbs.append(rgb)
            if max_points is not None and len(xyzs) >= max_points:
                break
    return xyzs, rgbs


def load_poster_scene(
    scene_name: str = "poster",
    num_cameras: int = 2,
    max_points: int = 8000,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> GaussianScene:
    scene_root = _download_scene(scene_name)
    width, height, K = _read_camera(scene_root / "cameras.bin")
    viewmats = torch.stack(_read_images(scene_root / "images.bin", limit=num_cameras), dim=0)
    xyzs, rgbs = _read_points(scene_root / "points3D.bin", max_points=max_points)

    means = torch.tensor(xyzs, device=device, dtype=dtype)
    colors = torch.tensor(rgbs, device=device, dtype=dtype) / 255.0
    opacities = torch.full((means.shape[0],), 0.7, device=device, dtype=dtype)
    scales = torch.full((means.shape[0], 3), 0.02, device=device, dtype=dtype)
    quats = torch.zeros((means.shape[0], 4), device=device, dtype=dtype)
    quats[:, 0] = 1.0
    Ks = K.to(device=device, dtype=dtype).unsqueeze(0).expand(viewmats.shape[0], -1, -1).contiguous()
    viewmats = viewmats.to(device=device, dtype=dtype)

    return GaussianScene(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        scene_name=scene_name,
    )


class GSplatReference(torch.nn.Module):
    def __init__(self, scene: GaussianScene):
        super().__init__()
        self.scene = scene

    def forward(self, return_meta: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict]:
        return rasterization(
            self.scene.means,
            self.scene.quats,
            self.scene.scales,
            self.scene.opacities,
            self.scene.colors,
            self.scene.viewmats,
            self.scene.Ks,
            self.scene.width,
            self.scene.height,
            packed=False,
        )


def load_reference_3dgs(scene: GaussianScene):
    return GSplatReference(scene).eval()


def load_ours_3dgs(scene: GaussianScene):
    from tasks.baseline.L4.gaussian_splatting import GaussianSplatting

    return GaussianSplatting(
        means=scene.means,
        quats=scene.quats,
        scales=scene.scales,
        opacities=scene.opacities,
        colors=scene.colors,
        width=scene.width,
        height=scene.height,
        viewmats=scene.viewmats,
        Ks=scene.Ks,
    ).eval()
