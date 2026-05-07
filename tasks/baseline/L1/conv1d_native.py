"""General-purpose Conv1d L1 wrapper with full torch.nn.Conv1d kwarg coverage.

Additive to the existing narrow ``tasks/baseline/L1/conv1d.py`` (which is a
Whisper-specific wrapper without `groups`/`dilation`/`padding_mode` and with
its weight nested under ``self.conv``). This wrapper:

- Stores ``weight`` and ``bias`` as direct ``nn.Parameter`` attributes so HF
  reference state_dicts (which use the keys ``weight`` and ``bias`` for
  ``nn.Conv1d``) load with no remapping.
- Supports the full kwarg surface: stride, padding, dilation, groups,
  padding_mode, bias.

This wrapper is what audited HF models with ``nn.Conv1d(..., groups=N)``
(granite_speech, vibevoice, squeezebert) or ``dilation=D`` (dac, encodec,
pe_audio) need.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dNative(nn.Module):
    """General Conv1d. Forward dispatches to F.conv1d (or pad+F.conv1d for
    ``padding_mode != 'zeros'``)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")
        if padding_mode not in ("zeros", "reflect", "replicate", "circular"):
            raise ValueError(f"unsupported padding_mode: {padding_mode}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            # Match torch.nn.Conv1d's reverse padding tuple convention
            x = F.pad(x, (self.padding, self.padding), mode=self.padding_mode)
            return F.conv1d(
                x, self.weight, self.bias,
                stride=self.stride, padding=0,
                dilation=self.dilation, groups=self.groups,
            )
        return F.conv1d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
        )
