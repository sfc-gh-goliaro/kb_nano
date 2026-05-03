"""Semantic PyTorch reference for variable-length Flash Attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from ._attention import varlen_attention


def _varlen_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    batch = cu_seqlens_q.numel() - 1
    num_heads = q.shape[1]
    max_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()) if batch else 0
    lse = torch.full((batch, num_heads, max_q), -float("inf"), dtype=torch.float32, device=q.device)
    for b in range(batch):
        qs = int(cu_seqlens_q[b].item())
        qe = int(cu_seqlens_q[b + 1].item())
        ks = int(cu_seqlens_k[b].item())
        ke = int(cu_seqlens_k[b + 1].item())
        q_b = q[qs:qe].float().transpose(0, 1)
        k_b = k[ks:ke].float().transpose(0, 1)
        scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * softmax_scale
        if causal:
            sq = qe - qs
            sk = ke - ks
            q_pos = torch.arange(sq, device=q.device) + max(sk - sq, 0)
            k_pos = torch.arange(sk, device=q.device)
            mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
            scores = scores.masked_fill(mask.unsqueeze(0), -float("inf"))
        lse[b, :, : qe - qs] = torch.logsumexp(scores, dim=-1)
    return lse


class FlashAttnVarlen(nn.Module):
    """Variable-length attention without paged KV cache lookup."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool = True,
        return_softmax_lse: bool = False,
    ):
        del max_seqlen_q, max_seqlen_k
        out = varlen_attention(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale,
            causal,
        )
        if not return_softmax_lse:
            return out
        return out, _varlen_lse(q, k, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal)
