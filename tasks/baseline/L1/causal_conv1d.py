"""Depthwise causal 1D convolution (L1).

Wraps ``fla.modules.convolution.causal_conv1d`` so that L2 callers (KDA,
GDN linear attention) never import the FLA library directly.

Interface mirrors the FLA function:
  forward(x, weight, *, initial_state=None, output_final_state=False,
          activation='silu', backend=None) -> (output, final_state)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fla.modules.convolution import causal_conv1d as _fla_causal_conv1d


class CausalConv1d(nn.Module):
    """Causal depthwise 1D conv with optional SiLU activation and stateful decode.

    Inputs:
      x:               [B, T, D]  pre-conv channel-major activations
      weight:          [D, K]     depthwise conv kernel
      initial_state:   [B, D, K-1] or None
      output_final_state: bool — when True, returns updated state for decode
      activation:      'silu' | 'swish' | None
      backend:         passed straight through to FLA (e.g. 'cuda')

    Returns: (output [B, T, D], final_state or None)
    """

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        activation: str | None = "silu",
        backend: str | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        kwargs = dict(
            activation=activation,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        if backend is not None:
            kwargs["backend"] = backend
        if cu_seqlens is not None:
            kwargs["cu_seqlens"] = cu_seqlens
        return _fla_causal_conv1d(x, weight, **kwargs)
