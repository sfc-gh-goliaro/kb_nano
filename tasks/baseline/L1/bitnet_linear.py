"""BitLinear: linear layer with per-token int8 activation quantization.

Core primitive for BitNet 1.58-bit models. Weights are stored in full
precision but expected to contain (approximately) ternary values
({-1, 0, +1} scaled by a per-tensor factor).

During forward pass, activations are quantized to int8 range and
immediately dequantized, adding quantization noise that matches the
BitNet training procedure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Per-token quantization to int8 range with immediate dequantization.

    Args:
        x: input tensor [..., D]

    Returns:
        Tensor of same shape with quantization noise applied.
    """
    s = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    return (x * s).round().clamp_(-128, 127) / s


class BitLinear(nn.Linear):
    """Linear layer with per-token int8 activation quantization.

    Inherits from nn.Linear so weight names match standard checkpoint
    conventions (e.g. ``self_attn.q_proj.weight``).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(activation_quant(x), self.weight)
