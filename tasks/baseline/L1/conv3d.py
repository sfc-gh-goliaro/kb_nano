"""Conv3d wrapper around nn.Conv3d.

Originally narrow (in_channels, out_channels, kernel_size, stride, bias) for
vision-encoder patch-embedding only. Extended additively to accept the full
nn.Conv3d kwarg surface — ``padding``, ``dilation``, ``groups``, ``padding_mode``
— that emu3's VQVAE temporal block uses (``nn.Conv3d(..., padding=0)``).

Defaults match torch.nn.Conv3d (padding=0, dilation=1, groups=1, bias=False
preserved for vllm-compat) so existing callers (Qwen2-VL, Qwen3-VL, video
patch-embed) continue to work unchanged.

Internal layout: ``self.conv = nn.Conv3d(...)`` is preserved so kb-nano
callers that access ``self.conv.weight`` keep working.
"""

from __future__ import annotations

import torch.nn as nn


class Conv3d(nn.Module):
    """Conv3D wrapper matching vllm's Conv3dLayer interface."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...] | int,
        stride: tuple[int, ...] | int | None = None,
        padding: tuple[int, ...] | int = 0,
        dilation: tuple[int, ...] | int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride or kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    @property
    def stride(self):
        return self.conv.stride

    @property
    def padding(self):
        return self.conv.padding

    def forward(self, x):
        return self.conv(x)
