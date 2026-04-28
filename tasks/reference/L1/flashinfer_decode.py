"""Semantic PyTorch reference for flashinfer_decode.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.

Limitations: FlashInfer/TRTLLM paged-cache execution is decomposed into a
Python gather plus PyTorch SDPA. Workspace and Blackwell-specific execution
details are intentionally ignored.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kb_nano.tasks.reference.L1._attention import dense_attention, gather_paged_cache


class TRTLLMDecode(nn.Module):
    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        workspace: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        self._workspace = workspace

    def forward(self, q, k_cache, v_cache, cache_seqlens=None,
                block_table=None, softmax_scale=None, causal=True,
                max_seq_len=None, **kwargs):
        del causal, max_seq_len, kwargs
        if cache_seqlens is None:
            cache_seqlens = torch.full((q.shape[0],), k_cache.shape[2], device=q.device, dtype=torch.int32)
        scale = softmax_scale if softmax_scale is not None else self.sm_scale
        outs = []
        for i in range(q.shape[0]):
            seq_len = int(cache_seqlens[i].item())
            k = gather_paged_cache(k_cache, block_table, i, seq_len, hnd=True)
            v = gather_paged_cache(v_cache, block_table, i, seq_len, hnd=True)
            out = dense_attention(
                q[i:i + 1].unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                softmax_scale=scale, causal=False,
            ).squeeze(0).squeeze(0)
            outs.append(out)
        return torch.stack(outs, dim=0)
