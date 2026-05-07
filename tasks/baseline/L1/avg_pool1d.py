"""AvgPool1d wrapping F.avg_pool1d directly.

Explicit L1 op (NOT a 2D-composed workaround). Per mentor guidance: composing
1D pool via 2D pool with degenerate H=1 benchmarks the wrong kernel family
and adds reshape overhead. This wrapper dispatches directly to the 1D kernel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgPool1d(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )
