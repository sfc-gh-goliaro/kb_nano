"""DP3 1-D conv block: Conv1d -> GroupNorm -> Mish.

Mirrors ``diffusion_policy_3d.model.diffusion.conv1d_components.Conv1dBlock``
exactly, including the same ``block`` Sequential layout so checkpoint keys
load without remapping. Uses ``torch.nn.Conv1d`` / ``ConvTranspose1d``
directly (the same pragmatic pattern as ``L2/cosyvoice3_hifigan.py``) to
preserve the ``block.0.weight`` / ``conv.weight`` parameter names from the
reference state_dict.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.group_norm import GroupNorm
from ..L1.mish import Mish


class Conv1dBlock(nn.Module):
    """Conv1d (kernel_size, padding=ks//2) -> GroupNorm(n_groups) -> Mish."""

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 8,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size,
                padding=kernel_size // 2,
            ),
            # eps=1e-5 matches torch.nn.GroupNorm default (the reference
            # uses nn.GroupNorm directly); kb-nano's L1 GroupNorm defaults
            # to eps=1e-6.
            GroupNorm(n_groups, out_channels, eps=1e-5),
            Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    """Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
