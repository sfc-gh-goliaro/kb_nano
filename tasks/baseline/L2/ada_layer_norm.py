"""Adaptive Layer Norm modules for diffusion transformers (L2 composite).

AdaLayerNormZero: 6-output adaLN-Zero for dual-stream FLUX blocks.
AdaLayerNormZeroSingle: 3-output adaLN-Zero for single-stream FLUX blocks.

Mirrors the diffusers ``AdaLayerNormZero`` / ``AdaLayerNormZeroSingle``
implementations, rebuilt on top of L1 ops.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU


class AdaLayerNormZero(nn.Module):
    """Adaptive layer norm zero (adaLN-Zero) for dual-stream DiT blocks.

    Projects the timestep embedding through SiLU + Linear into 6 modulation
    signals (shift/scale/gate for MSA and MLP), then applies LayerNorm with
    adaptive scale and shift.
    """

    def __init__(self, embedding_dim: int, norm_type: str = "layer_norm", bias: bool = True):
        super().__init__()
        self.silu = SiLU()
        self.linear = Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        self.norm = LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    """Adaptive layer norm zero (adaLN-Zero) for single-stream DiT blocks.

    Projects the timestep embedding through SiLU + Linear into 3 modulation
    signals (shift/scale/gate for the fused MSA+MLP path).
    """

    def __init__(self, embedding_dim: int, norm_type: str = "layer_norm", bias: bool = True):
        super().__init__()
        self.silu = SiLU()
        self.linear = Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        self.norm = LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa
