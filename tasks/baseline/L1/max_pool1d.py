"""MaxPool1d wrapping F.max_pool1d directly.

Explicit L1 op (NOT a 2D-composed workaround). Per mentor guidance: composing
1D pool as `MaxPool2d((1, k))(x.unsqueeze(-2)).squeeze(-2)` is functionally
equivalent but benchmarks the wrong kernel family (2D pool with degenerate
H=1) and adds reshape overhead. This wrapper dispatches directly to the 1D
kernel.
"""

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
        return_indices: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor):
        return F.max_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
