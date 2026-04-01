"""Diffusion conditioning for AlphaFold3.

Produces conditioned single and pair representations from trunk outputs
and diffusion time step. Implements Fourier time embedding and optional
trunk conditioning.

Reference: openfold3/core/model/layers/diffusion_conditioning.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


class FourierEmbedding(nn.Module):
    """Fourier time embedding for diffusion conditioning.

    Args:
        c_out: Output dimension
    """

    def __init__(self, c_out: int):
        super().__init__()
        self.c_out = c_out
        self.linear = Linear(c_out, c_out, bias=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [*] time values

        Returns:
            [*, c_out] Fourier embedding
        """
        half_dim = self.c_out // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        args = t[..., None] * freq
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.c_out % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return self.linear(embedding)


class DiffusionConditioning(nn.Module):
    """Conditioning for diffusion module.

    Combines trunk single/pair representations with time embedding.

    Args:
        c_s: Single representation channel dimension
        c_z: Pair representation channel dimension
        c_s_input: Input single representation dimension
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_input: int = 449,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z

        self.fourier_embedding = FourierEmbedding(c_s)
        self.layer_norm_s = LayerNorm(c_s)
        self.linear_s = Linear(c_s, c_s, bias=False)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, c_z, bias=False)
        self.linear_s_input = Linear(c_s_input, c_s, bias=False)

    def forward(
        self,
        batch: dict,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        use_conditioning: bool,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:     Feature dictionary
            t:         [*] noise level
            si_input:  [*, N_token, c_s_input] input embedding
            si_trunk:  [*, N_token, c_s] trunk single rep
            zij_trunk: [*, N_token, N_token, c_z] trunk pair rep
            use_conditioning: Whether to condition with trunk reps

        Returns:
            si:  [*, N_token, c_s] conditioned single rep
            zij: [*, N_token, N_token, c_z] conditioned pair rep
        """
        t_emb = self.fourier_embedding(t)

        si = self.linear_s_input(si_input) + t_emb[..., None, :]

        if use_conditioning:
            si = si + self.linear_s(self.layer_norm_s(si_trunk))

        zij = zij_trunk
        if use_conditioning:
            zij = self.linear_z(self.layer_norm_z(zij_trunk))

        return si, zij
