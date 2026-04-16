"""Grouped top-k routing for DeepSeek-style MoE.

Implements sigmoid-gated grouped-top-k with optional bias, matching
vLLM's ``grouped_topk`` helper and sgl_kernel's ``moe_fused_gate``.  This
is the pure-PyTorch reference used by DeepSeek V3: scores are obtained via
``sigmoid(router_logits)``, groups are scored by the sum of their top-2
experts, the top ``topk_group`` groups are selected, and top-``topk``
experts are then picked from within those groups.

Normalisation matches vLLM's pattern: if ``topk > 1`` the kept weights are
renormalised to sum to 1 before returning.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GroupedTopK(nn.Module):
    def forward(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor | None,
        num_expert_group: int,
        topk_group: int,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(router_logits.float())
        num_tokens, num_experts = scores.shape

        if bias is not None:
            scores_for_choice = scores + bias.float()
        else:
            scores_for_choice = scores

        experts_per_group = num_experts // num_expert_group
        grouped = scores_for_choice.view(num_tokens, num_expert_group, experts_per_group)
        group_score = grouped.topk(2, dim=-1).values.sum(dim=-1)
        top_group_idx = group_score.topk(topk_group, dim=-1, sorted=False).indices

        group_mask = torch.zeros_like(group_score)
        group_mask.scatter_(1, top_group_idx, 1.0)
        expert_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, num_expert_group, experts_per_group)
            .reshape(num_tokens, num_experts)
        )
        masked_scores = scores_for_choice.masked_fill(expert_mask == 0, float("-inf"))
        topk_ids = masked_scores.topk(topk, dim=-1, sorted=False).indices
        topk_weights = scores.gather(1, topk_ids)

        if topk > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        return topk_weights.to(router_logits.dtype), topk_ids.to(torch.int32)
