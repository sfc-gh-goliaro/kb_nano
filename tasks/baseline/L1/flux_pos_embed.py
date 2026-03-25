"""2D rotary position embeddings for FLUX (concatenated per-axis 1D embeddings).

Generates (cos, sin) tensors from integer grid IDs for use with DiffusionRoPE.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    freqs_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute 1D rotary frequency tensor as complex exponentials.

    Returns complex64 tensor of shape [S, dim/2].
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
    freqs = torch.outer(pos, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


class FluxPosEmbed(nn.Module):
    """2D rotary position embeddings for FLUX."""

    def __init__(self, theta: int, axes_dim: list[int] | tuple[int, ...]):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        for i in range(n_axes):
            freqs_cis = _get_1d_rotary_pos_embed(
                self.axes_dim[i], pos[:, i],
                theta=self.theta, freqs_dtype=torch.float64,
            )
            cos_out.append(freqs_cis.real)
            sin_out.append(freqs_cis.imag)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin
