"""ConvTranspose3d parametric op wrapping F.conv_transpose3d.

Stores ``weight`` and ``bias`` as direct ``nn.Parameter`` attributes so
reference state_dicts (HF / torch.nn.ConvTranspose3d) load with no remapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTranspose3d(nn.Module):
    """Parametric 3-D transposed convolution. Forward dispatches to
    ``F.conv_transpose3d``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        output_padding: int | tuple[int, int, int] = 0,
        groups: int = 1,
        dilation: int | tuple[int, int, int] = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding, output_padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, *kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )
