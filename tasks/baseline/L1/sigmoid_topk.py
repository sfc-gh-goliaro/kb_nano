"""Sigmoid top-k routing for Mixture-of-Experts.

Includes both simple top-k (Llama 4) and grouped top-k (DeepSeek V3)
with optional noaux_tc e_score_correction_bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SigmoidTopK(nn.Module):
    """Top-k selection with sigmoid weights (Llama 4 style)."""

    def __init__(self):
        super().__init__()
        self._topk_weights = None
        self._topk_ids = None

    def _ensure_buffers(self, M, top_k, device):
        if self._topk_weights is None or self._topk_weights.size(0) < M:
            self._topk_weights = torch.empty(
                M, top_k, device=device, dtype=torch.float32,
            )
            self._topk_ids = torch.empty(
                M, top_k, device=device, dtype=torch.int32,
            )

    def forward(
        self,
        router_logits: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M = router_logits.size(0)
        self._ensure_buffers(M, top_k, router_logits.device)
        topk_weights = self._topk_weights[:M]
        topk_ids = self._topk_ids[:M]
        scores, indices = torch.topk(router_logits, top_k, dim=-1)
        topk_weights.copy_(torch.sigmoid(scores.float()))
        topk_ids.copy_(indices.to(torch.int32))
        return topk_weights, topk_ids


class GroupedSigmoidTopK(nn.Module):
    """DeepSeek V3-style grouped top-k with sigmoid scoring.

    Algorithm:
      1. Compute sigmoid scores for all experts
      2. Add e_score_correction_bias (noaux_tc) to biased scores
      3. Group experts into n_group groups of (n_experts // n_group)
      4. Score each group by the sum of its top-2 biased expert scores
      5. Select top topk_group groups
      6. Within selected groups, pick top-k experts total using biased scores
      7. Return original (unbiased) sigmoid weights for the chosen experts
      8. Multiply by routed_scaling_factor
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        router_logits: torch.Tensor,
        top_k: int,
        n_group: int,
        topk_group: int,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float = 1.0,
        renormalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = router_logits.sigmoid()

        num_token = scores.size(0)
        num_experts = scores.size(1)

        if e_score_correction_bias is not None:
            original_scores = scores
            scores = scores + e_score_correction_bias.unsqueeze(0)

            group_scores = (
                scores.view(num_token, n_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
        else:
            original_scores = scores
            group_scores = (
                scores.view(num_token, n_group, -1).max(dim=-1).values
            )

        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_token, n_group, num_experts // n_group)
            .reshape(num_token, -1)
        )
        tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

        if e_score_correction_bias is not None:
            topk_ids = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)[1]
            topk_weights = original_scores.gather(1, topk_ids)
        else:
            topk_weights, topk_ids = torch.topk(
                tmp_scores, k=top_k, dim=-1, sorted=False,
            )

        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * routed_scaling_factor

        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)
