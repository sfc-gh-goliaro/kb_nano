"""Patch merger for Qwen vision encoders.

Merges spatial_merge_size^2 patches into one via norm + 2-layer MLP.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from .parallel_linear import ColumnParallelLinear, RowParallelLinear


class VisionPatchMerger(nn.Module):
    """Qwen2-VL style patch merger: ln_q -> flatten -> MLP(fc1, GELU, fc2)."""

    def __init__(self, d_model: int, context_dim: int,
                 spatial_merge_size: int = 2, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = nn.LayerNorm(context_dim, eps=eps)
        self.fc1 = ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True)
        self.act = nn.GELU()
        self.fc2 = RowParallelLinear(self.hidden_size, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)
        x = self.fc2(self.act(self.fc1(x)))
        return x


class Qwen3VisionPatchMerger(nn.Module):
    """Qwen3-VL style patch merger with optional postshuffle norm."""

    def __init__(self, d_model: int, context_dim: int,
                 spatial_merge_size: int = 2,
                 use_postshuffle_norm: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        self.norm = nn.LayerNorm(norm_dim, eps=eps)
        self.fc1 = ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True)
        self.act = nn.GELU()
        self.fc2 = RowParallelLinear(self.hidden_size, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        x = self.fc2(self.act(self.fc1(x)))
        return x
