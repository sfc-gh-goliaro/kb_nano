"""EfficientNetV2 building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.silu import SiLU


class BatchNormAct2d(BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        act_layer: nn.Module | None = None,
    ):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
        self.drop = nn.Identity()
        self.act = act_layer if act_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = self.drop(x)
        return self.act(x)


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels: int, reduced_channels: int):
        super().__init__()
        self.conv_reduce = Conv2d(in_channels, reduced_channels, kernel_size=1, bias=True)
        self.act1 = SiLU()
        self.conv_expand = Conv2d(reduced_channels, in_channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(-2, -1), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.gate(scale)
        return x * scale


class EdgeResidual(nn.Module):
    def __init__(self, in_chs: int, exp_chs: int, out_chs: int, stride: int, has_skip: bool):
        super().__init__()
        self.has_skip = has_skip
        self.conv_exp = Conv2d(in_chs, exp_chs, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNormAct2d(exp_chs, act_layer=SiLU())
        self.aa = nn.Identity()
        self.se = nn.Identity()
        self.conv_pwl = Conv2d(exp_chs, out_chs, kernel_size=1, bias=False)
        self.bn2 = BatchNormAct2d(out_chs, act_layer=nn.Identity())
        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.aa(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)
        if self.has_skip:
            x = residual + self.drop_path(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_chs: int,
        exp_chs: int,
        out_chs: int,
        stride: int,
        se_reduce_chs: int,
        has_skip: bool,
    ):
        super().__init__()
        self.has_skip = has_skip
        self.conv_pw = Conv2d(in_chs, exp_chs, kernel_size=1, bias=False)
        self.bn1 = BatchNormAct2d(exp_chs, act_layer=SiLU())
        self.conv_dw = Conv2d(exp_chs, exp_chs, kernel_size=3, stride=stride, padding=1, groups=exp_chs, bias=False)
        self.bn2 = BatchNormAct2d(exp_chs, act_layer=SiLU())
        self.aa = nn.Identity()
        self.se = SqueezeExcite(exp_chs, se_reduce_chs)
        self.conv_pwl = Conv2d(exp_chs, out_chs, kernel_size=1, bias=False)
        self.bn3 = BatchNormAct2d(out_chs, act_layer=nn.Identity())
        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.aa(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            x = residual + self.drop_path(x)
        return x
