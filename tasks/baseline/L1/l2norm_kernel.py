"""Fused L2-norm forward kernel (L1).

Wraps vLLM's bundled FLA ``l2norm_fwd`` Triton kernel. Used by the GDN
linear-attention prefill path where we need bitwise alignment with
vLLM's normalization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.model_executor.layers.fla.ops.l2norm import (
    l2norm_fwd as _vllm_l2norm_fwd,
)


class L2NormFwd(nn.Module):
    """Fused Triton L2-norm matching vLLM's numerics."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _vllm_l2norm_fwd(x)
