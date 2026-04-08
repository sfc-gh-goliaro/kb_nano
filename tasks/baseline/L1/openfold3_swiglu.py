"""SwiGLU activation and transition for AlphaFold3.

SwiGLU: SiLU(linear_a(x)) * linear_b(x)
SwiGLUTransition: LayerNorm -> SwiGLU -> Linear (AF3 Algorithm 11)
ConditionedTransitionBlock: AdaLN -> SwiGLU -> gated output (AF3 Algorithm 25)

Reference: openfold3/core/model/primitives/activations.py SwiGLU
           openfold3/core/model/primitives/normalization.py AdaLN
           openfold3/core/model/layers/transition.py SwiGLUTransition
           openfold3/core/model/layers/transition.py ConditionedTransitionBlock
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layer_norm import LayerNorm
from .linear import Linear


class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(Wa x) * Wb x.

    Args:
        c_in: Number of input channels
        c_out: Number of output channels
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.linear_a = Linear(c_in, c_out, bias=False)
        self.linear_b = Linear(c_in, c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.linear_a(x)) * self.linear_b(x)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization matching the reference AdaLN.

    Submodule structure matches checkpoint keys:
    - layer_norm_s: LayerNorm(c_s), weight-only
    - linear_g: Linear(c_s, c_a, bias=True) — gating
    - linear_s: Linear(c_s, c_a, bias=False) — additive conditioning

    Reference: openfold3/core/model/primitives/normalization.py AdaLN

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
    """

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s

        self.layer_norm_a = LayerNorm(c_a, create_scale=False, create_offset=False)
        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.sigmoid = nn.Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_s = Linear(c_s, c_a, bias=False)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s_norm = self.layer_norm_s(s)
        g = self.sigmoid(self.linear_g(s_norm))
        a_norm = self.layer_norm_a(a)
        return g * (a_norm + self.linear_s(s_norm))


class SwiGLUTransition(nn.Module):
    """AF3 Algorithm 11: SwiGLU-based transition.

    Args:
        c_in: Input channel dimension
        n: Factor multiplied to c_in for hidden dimension
    """

    def __init__(self, c_in: int, n: int):
        super().__init__()
        self.c_in = c_in
        self.n = n

        self.layer_norm = LayerNorm(c_in)
        self.swiglu = SwiGLU(c_in, n * c_in)
        self.linear_out = Linear(n * c_in, c_in, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        ckpt_chunk_size: int | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        mask = mask.unsqueeze(-1)

        x = self.layer_norm(x)
        x = self.swiglu(x)
        x = self.linear_out(x)
        x = x * mask

        return x


class ConditionedTransitionBlock(nn.Module):
    """AF3 Algorithm 25: SwiGLU transition with AdaLN-Zero conditioning.

    Submodule names match the reference checkpoint:
    - layer_norm: AdaLN
    - swiglu: SwiGLU
    - linear_g: output gate Linear(c_s, c_a, bias=True)
    - linear_out: Linear(n*c_a, c_a, bias=False)

    Reference: openfold3/core/model/layers/transition.py ConditionedTransitionBlock

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
        n: Factor for hidden dimension
    """

    def __init__(self, c_a: int, c_s: int, n: int):
        super().__init__()
        self.layer_norm = AdaLN(c_a=c_a, c_s=c_s)
        self.swiglu = SwiGLU(c_a, n * c_a)
        self.sigmoid = nn.Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_out = Linear(n * c_a, c_a, bias=False)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        mask = mask.unsqueeze(-1)

        a = self.layer_norm(a, s)
        b = self.swiglu(a)
        a = self.sigmoid(self.linear_g(s)) * self.linear_out(b)
        a = a * mask

        return a
