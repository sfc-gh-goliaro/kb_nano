"""MaxPool1d wrapping F.max_pool1d."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool1d(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        dilation: int = 1,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )
