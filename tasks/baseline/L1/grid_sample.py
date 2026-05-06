"""GridSample wrapping F.grid_sample."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridSample(nn.Module):
    def __init__(
        self,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            input,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
