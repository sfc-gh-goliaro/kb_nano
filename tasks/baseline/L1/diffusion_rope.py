"""Rotary position embedding for diffusion models (interleaved / GPT-J style).

Unlike the LLM RoPE (which uses sgl_kernel with a precomputed cos/sin cache
indexed by integer positions), diffusion models receive pre-computed (cos, sin)
tensors of shape [seq_len, rotary_dim/2] and apply them to Q/K of shape
[batch, seq_len, num_heads, head_dim].
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply rotary embeddings.  x: (B, S, H, D), cos/sin: (S, D/2)."""
    ro_dim = cos.shape[-1] * 2
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    if interleaved:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
    else:
        repeat_dims = [1] * cos.dim()
        repeat_dims[-1] = 2
        cos = cos.repeat(*repeat_dims)
        sin = sin.repeat(*repeat_dims)
    return torch.cat(
        [
            x[..., :ro_dim] * cos + _rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class DiffusionRoPE(nn.Module):
    """Apply rotary embeddings given pre-computed (cos, sin) tensors.

    Parameters
    ----------
    is_neox_style : bool
        If True, use the GPT-NeoX (half-split) layout.
        If False (default for FLUX), use the interleaved (GPT-J) layout.
    """

    def __init__(self, is_neox_style: bool = False) -> None:
        super().__init__()
        self.is_neox_style = is_neox_style
        self.interleaved = not is_neox_style

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if cos.dim() == 3:
            cos = cos[0]
            sin = sin[0]

        return _apply_rotary_emb(x, cos, sin, interleaved=self.interleaved)
