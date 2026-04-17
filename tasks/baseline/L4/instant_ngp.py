"""Repo-native InstantNGP renderer built from snapshot assets and tinycudann kernels."""

from __future__ import annotations

import torch
import torch.nn as nn

from infra.nerf_loader import InstantNGPSnapshotAssets
from tasks.baseline.L2.instantngp_field import InstantNGPField
from tasks.baseline.L3.instantngp_renderer import InstantNGPRenderer


class InstantNGP(nn.Module):
    def __init__(
        self,
        *,
        assets: InstantNGPSnapshotAssets,
        width: int,
        height: int,
        spp: int = 1,
        scene_name: str = "fox",
        ray_chunk_size: int = 131072,
        use_exact_marcher: bool = True,
        render_step_scale: float = 0.5,
        fast_alpha_thre: float = 0.0,
        fast_early_stop_eps: float = 0.0,
        fast_use_sigma_pruning: bool = False,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.spp = spp
        self.scene_name = scene_name

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.field = InstantNGPField(
            hashgrid_config=assets.hashgrid_config,
            direction_encoding_config=assets.direction_encoding_config,
            density_network_config=assets.density_network_config,
            rgb_network_config=assets.rgb_network_config,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        self.field.load_flat_params(assets.flat_params.to(device))

        self.renderer = InstantNGPRenderer(
            field=self.field,
            assets=assets,
            width=width,
            height=height,
            spp=spp,
            ray_chunk_size=ray_chunk_size,
            use_exact_marcher=use_exact_marcher,
            render_step_scale=render_step_scale,
            fast_alpha_thre=fast_alpha_thre,
            fast_early_stop_eps=fast_early_stop_eps,
            fast_use_sigma_pruning=fast_use_sigma_pruning,
        ).to(device)

    def render(
        self,
        view_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        spp: int | None = None,
        linear: bool = True,
    ) -> torch.Tensor:
        if not linear:
            raise ValueError("Native InstantNGP renderer currently only supports linear=True")
        if spp is not None and spp != self.spp:
            raise ValueError("Native InstantNGP renderer currently assumes a fixed spp")
        output = self.renderer.render(
            view_index=view_index,
            width=width or self.width,
            height=height or self.height,
        )
        return output.rgba

    def forward(
        self,
        view_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        spp: int | None = None,
    ) -> torch.Tensor:
        return self.render(
            view_index=view_index,
            width=width,
            height=height,
            spp=spp,
            linear=True,
        )
