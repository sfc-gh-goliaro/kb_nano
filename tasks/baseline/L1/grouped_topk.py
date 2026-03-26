"""Grouped top-k routing for DeepSeek MoE via sgl_kernel."""

from __future__ import annotations

import torch
import torch.nn as nn

from sgl_kernel.moe import moe_fused_gate as _sgl_fused_gate


class GroupedTopK(nn.Module):
    def forward(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor | None,
        num_expert_group: int,
        topk_group: int,
        topk: int,
        routed_scaling_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids = _sgl_fused_gate(
            router_logits,
            bias,
            num_expert_group,
            topk_group,
            topk,
        )
        topk_weights = topk_weights * routed_scaling_factor
        return topk_weights, topk_ids
