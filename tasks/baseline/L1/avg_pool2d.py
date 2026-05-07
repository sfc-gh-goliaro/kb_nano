"""AvgPool2d wrapping F.avg_pool2d.

Originally narrow (kernel_size, stride, padding, ceil_mode). Extended additively
with ``count_include_pad`` and ``divisor_override`` so HF models that pass
those kwargs (e.g. ``nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2,
count_include_pad=False)``) drop in without modification.

Defaults match torch.nn.AvgPool2d so existing kb-nano callers are unaffected.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )
