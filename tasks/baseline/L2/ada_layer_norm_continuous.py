"""Adaptive continuous layer norm for diffusion transformers (L2 composite).

Used as the final output norm in FLUX (``norm_out``).  Projects the
conditioning embedding through SiLU + Linear into per-channel scale and
shift, then applies LayerNorm with those modulations.

Mirrors diffusers' ``AdaLayerNormContinuous``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU


class AdaLayerNormContinuous(nn.Module):
    """Continuous adaptive layer norm.

    Parameters
    ----------
    embedding_dim : int
        Hidden dimension of the input.
    conditioning_embedding_dim : int
        Dimension of the conditioning (timestep) embedding.
    elementwise_affine : bool
        Whether the inner LayerNorm has learnable affine parameters.
    eps : float
        Epsilon for LayerNorm.
    bias : bool
        Whether the projection Linear has bias.
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        self.silu = SiLU()
        self.linear = Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        self.norm = LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
