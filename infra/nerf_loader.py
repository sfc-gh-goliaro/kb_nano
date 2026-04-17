"""Loaders and utilities for InstantNGP-style NeRF benchmarks."""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import msgpack
import numpy as np
import torch
import torch.nn as nn


_KB_ROOT = Path(__file__).resolve().parents[1]
_INSTANT_NGP_ROOT = _KB_ROOT / "third_party" / "instant-ngp"
_INSTANT_NGP_BUILD = _INSTANT_NGP_ROOT / "build"
_SNAPSHOT_CACHE = Path.home() / ".cache" / "kb_nano" / "instantngp"


@dataclass
class InstantNGPScene:
    scene_name: str
    scene_path: str
    width: int
    height: int
    num_views: int


@dataclass
class InstantNGPView:
    camera_matrix: torch.Tensor
    focal_length: torch.Tensor
    principal_point: torch.Tensor
    resolution: tuple[int, int]
    lens_params: torch.Tensor


@dataclass
class InstantNGPSnapshotAssets:
    scene_name: str
    views: list[InstantNGPView]
    hashgrid_config: dict
    direction_encoding_config: dict
    density_network_config: dict
    rgb_network_config: dict
    flat_params: torch.Tensor
    density_grid: torch.Tensor
    aabb_min: torch.Tensor
    aabb_max: torch.Tensor
    render_aabb_min: torch.Tensor
    render_aabb_max: torch.Tensor
    render_aabb_to_local: torch.Tensor
    background_color: torch.Tensor
    max_cascade: int
    cone_angle_constant: float
    min_transmittance: float
    train_in_linear_colors: bool
    render_near_distance: float
    snap_to_pixel_centers: bool


def is_instantngp_model(model_name: str) -> bool:
    name = model_name.lower()
    return "instantngp" in name or "instant-ngp" in name


def _ensure_pyngp():
    build = str(_INSTANT_NGP_BUILD.resolve())
    scripts = str((_INSTANT_NGP_ROOT / "scripts").resolve())
    if build not in sys.path:
        sys.path.insert(0, build)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import pyngp as ngp  # type: ignore

    return ngp


def load_fox_scene(scene_name: str = "fox") -> InstantNGPScene:
    scene_root = _INSTANT_NGP_ROOT / "data" / "nerf" / scene_name
    transforms_path = scene_root / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(
            f"InstantNGP scene metadata not found at {transforms_path}"
        )
    obj = json.loads(transforms_path.read_text())
    return InstantNGPScene(
        scene_name=scene_name,
        scene_path=str(scene_root.resolve()),
        width=int(obj["w"]),
        height=int(obj["h"]),
        num_views=len(obj["frames"]),
    )


def ensure_fox_snapshot(
    train_steps: int = 50,
    scene_name: str = "fox",
) -> str:
    scene = load_fox_scene(scene_name)
    _SNAPSHOT_CACHE.mkdir(parents=True, exist_ok=True)
    snapshot_path = _SNAPSHOT_CACHE / f"{scene_name}_{train_steps}steps.ingp"
    if snapshot_path.is_file():
        return str(snapshot_path)

    ngp = _ensure_pyngp()
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.root_dir = str(_INSTANT_NGP_ROOT.resolve())
    testbed.load_training_data(scene.scene_path)
    testbed.shall_train = True
    for _ in range(train_steps):
        testbed.frame()
    testbed.save_snapshot(str(snapshot_path), False, True)
    return str(snapshot_path)


class InstantNGPReference(nn.Module):
    def __init__(
        self,
        testbed,
        width: int,
        height: int,
        spp: int = 1,
        linear: bool = True,
    ):
        super().__init__()
        self.testbed = testbed
        self.width = width
        self.height = height
        self.spp = spp
        self.linear = linear

    def forward(
        self,
        view_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        spp: int | None = None,
    ) -> torch.Tensor:
        self.testbed.set_camera_to_training_view(view_index)
        image = self.testbed.render(
            width or self.width,
            height or self.height,
            spp or self.spp,
            self.linear,
        )
        return torch.from_numpy(image.copy())


def load_reference_instantngp(
    scene_name: str = "fox",
    train_steps: int = 50,
    width: int | None = None,
    height: int | None = None,
    spp: int = 1,
):
    ngp = _ensure_pyngp()
    scene = load_fox_scene(scene_name)
    snapshot_path = ensure_fox_snapshot(train_steps=train_steps, scene_name=scene_name)
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.root_dir = str(_INSTANT_NGP_ROOT.resolve())
    testbed.load_training_data(scene.scene_path)
    testbed.load_snapshot(snapshot_path)
    testbed.shall_train = False
    return InstantNGPReference(
        testbed=testbed,
        width=width or scene.width,
        height=height or scene.height,
        spp=spp,
        linear=True,
    ).eval()


def _load_snapshot_msgpack(snapshot_path: str) -> dict:
    with gzip.open(snapshot_path, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False, strict_map_key=False)


def _derive_hashgrid_config(config: dict, aabb_scale: int) -> dict:
    cfg = dict(config)
    if "per_level_scale" not in cfg:
        n_levels = int(cfg["n_levels"])
        if n_levels > 1:
            desired_resolution = 2048.0
            cfg["per_level_scale"] = math.exp(
                math.log(desired_resolution * float(aabb_scale) / float(cfg["base_resolution"]))
                / float(n_levels - 1)
            )
    return cfg


def load_native_instantngp_assets(
    scene_name: str = "fox",
    train_steps: int = 50,
) -> InstantNGPSnapshotAssets:
    snapshot_path = ensure_fox_snapshot(train_steps=train_steps, scene_name=scene_name)
    obj = _load_snapshot_msgpack(snapshot_path)
    snapshot = obj["snapshot"]
    dataset = snapshot["nerf"]["dataset"]
    aabb_scale = int(snapshot["nerf"]["aabb_scale"])

    views: list[InstantNGPView] = []
    for xform, metadata in zip(dataset["xforms"], dataset["metadata"], strict=True):
        lens = metadata["lens"]
        views.append(
            InstantNGPView(
                camera_matrix=torch.tensor(xform["start"], dtype=torch.float32),
                focal_length=torch.tensor(metadata["focal_length"], dtype=torch.float32),
                principal_point=torch.tensor(metadata["principal_point"], dtype=torch.float32),
                resolution=(int(metadata["resolution"][0]), int(metadata["resolution"][1])),
                lens_params=torch.tensor(
                    [
                        float(lens.get("k1", 0.0)),
                        float(lens.get("k2", 0.0)),
                        float(lens.get("p1", 0.0)),
                        float(lens.get("p2", 0.0)),
                    ],
                    dtype=torch.float32,
                ),
            )
        )

    flat_params = torch.from_numpy(
        np.frombuffer(snapshot["params_binary"], dtype=np.float16).copy()
    ).to(torch.float32)
    density_grid = torch.from_numpy(
        np.frombuffer(snapshot["density_grid_binary"], dtype=np.float16).copy()
    ).to(torch.float32)
    grid_cells = 128 * 128 * 128
    max_cascade = density_grid.numel() // grid_cells - 1

    return InstantNGPSnapshotAssets(
        scene_name=scene_name,
        views=views,
        hashgrid_config=_derive_hashgrid_config(obj["encoding"], aabb_scale),
        direction_encoding_config=dict(obj["dir_encoding"]),
        density_network_config=dict(obj["network"]),
        rgb_network_config=dict(obj["rgb_network"]),
        flat_params=flat_params,
        density_grid=density_grid,
        aabb_min=torch.tensor(snapshot["aabb"]["min"], dtype=torch.float32),
        aabb_max=torch.tensor(snapshot["aabb"]["max"], dtype=torch.float32),
        render_aabb_min=torch.tensor(snapshot["render_aabb"]["min"], dtype=torch.float32),
        render_aabb_max=torch.tensor(snapshot["render_aabb"]["max"], dtype=torch.float32),
        render_aabb_to_local=torch.tensor(snapshot["render_aabb_to_local"], dtype=torch.float32),
        background_color=torch.tensor(snapshot["background_color"], dtype=torch.float32),
        max_cascade=max_cascade,
        cone_angle_constant=0.0 if aabb_scale <= 1 else (1.0 / 256.0),
        min_transmittance=0.01,
        train_in_linear_colors=bool(dataset["is_hdr"]),
        render_near_distance=0.0,
        snap_to_pixel_centers=False,
    )


def load_ours_instantngp(
    scene_name: str = "fox",
    train_steps: int = 50,
    width: int | None = None,
    height: int | None = None,
    spp: int = 1,
    use_exact_marcher: bool = True,
    ray_chunk_size: int = 131072,
    render_step_scale: float = 0.5,
    fast_alpha_thre: float = 0.0,
    fast_early_stop_eps: float = 0.0,
    fast_use_sigma_pruning: bool = False,
):
    from tasks.baseline.L4.instant_ngp import InstantNGP

    assets = load_native_instantngp_assets(scene_name=scene_name, train_steps=train_steps)
    default_width, default_height = assets.views[0].resolution
    return InstantNGP(
        assets=assets,
        width=width or default_width,
        height=height or default_height,
        spp=spp,
        scene_name=scene_name,
        ray_chunk_size=ray_chunk_size,
        use_exact_marcher=use_exact_marcher,
        render_step_scale=render_step_scale,
        fast_alpha_thre=fast_alpha_thre,
        fast_early_stop_eps=fast_early_stop_eps,
        fast_use_sigma_pruning=fast_use_sigma_pruning,
    ).eval()
