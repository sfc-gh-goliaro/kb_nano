"""Chunk Gated Delta Rule — Triton-accelerated chunked kernel from FLA.

Thin ``nn.Module`` wrapper around ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``.
This is the FLA op used by Qwen3.5 / Qwen3-Next / OLMo-Hybrid for their
gated-delta-rule recurrence (a generalization of DeltaNet with per-head
gating). Mirrors the pattern of ``chunk_gla.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


class ChunkGatedDeltaRule(nn.Module):
    """Triton chunk gated-delta-rule kernel."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor | None = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )


class FusedRecurrentGatedDeltaRule(nn.Module):
    """FLA fused-recurrent gated-delta-rule decode kernel."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor | None = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return fused_recurrent_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
