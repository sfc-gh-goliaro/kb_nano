"""ConvNeXtV2 stage with optional downsampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.conv2d import Conv2d
from ..L1.layer_norm import LayerNorm
from ..L2.convnextv2_block import ConvNeXtV2Layer


class ConvNeXtV2LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ConvNeXtV2Stage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim: int,
        depth: int,
        hidden_dim: int,
        downsample: bool,
    ):
        super().__init__()
        if downsample:
            self.downsampling_layer = nn.ModuleList([
                ConvNeXtV2LayerNorm2d(in_dim, eps=1e-6),
                Conv2d(in_dim, dim, kernel_size=2, stride=2),
            ])
        else:
            self.downsampling_layer = nn.ModuleList()
        self.layers = nn.ModuleList([ConvNeXtV2Layer(dim, hidden_dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.downsampling_layer) > 0:
            x = self.downsampling_layer[0](x)
            x = self.downsampling_layer[1](x)
        for layer in self.layers:
            x = layer(x)
        return x
