"""Semantic PyTorch reference for fused top-k + softmax routing."""

from __future__ import annotations

import torch
import torch.nn as nn

class TopKSoftmax(nn.Module):
    """Top-k expert selection with softmax normalization."""

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
        renormalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k experts with softmax weights.

        Args:
            router_logits: [M, num_experts] router scores
            top_k: number of experts per token
            renormalize: renormalize weights to sum to 1

        Returns:
            topk_weights: [M, top_k] float32
            topk_ids: [M, top_k] int32
        """
        M = router_logits.size(0)
        self._ensure_buffers(M, top_k, router_logits.device)
        topk_weights = self._topk_weights[:M]
        topk_ids = self._topk_ids[:M]
        probs = torch.softmax(router_logits.float(), dim=-1)
        # The production kernel selects experts by k sequential argmax passes
        # with strict ">" while scanning ascending expert indices, so on tied
        # probabilities the LOWEST expert index wins.  bf16 logits tie often;
        # torch.topk's tie order is not the same, so mirror the sequential
        # argmax (torch.argmax also returns the first maximal index).
        work = probs.clone()
        ids_cols = []
        w_cols = []
        for _ in range(top_k):
            idx = work.argmax(dim=-1, keepdim=True)
            w_cols.append(probs.gather(-1, idx))
            ids_cols.append(idx)
            work.scatter_(-1, idx, float("-inf"))
        weights = torch.cat(w_cols, dim=-1)
        ids = torch.cat(ids_cols, dim=-1)
        if renormalize:
            # kernel: multiply by the reciprocal of the selected sum (no eps)
            weights = weights * (1.0 / weights.sum(dim=-1, keepdim=True))
        topk_weights.copy_(weights.to(topk_weights.dtype))
        topk_ids.copy_(ids.to(topk_ids.dtype))
        return topk_weights, topk_ids
