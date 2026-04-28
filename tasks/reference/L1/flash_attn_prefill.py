"""Semantic PyTorch reference for flash_attn_prefill.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kb_nano.tasks.reference.L1._attention import gather_paged_cache, varlen_attention


class FlashAttnPrefill(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        del max_seqlen_q, max_seqlen_k
        block_table = kwargs.get("block_table")
        window_size = kwargs.get("window_size", (-1, -1))
        window_size = (-1, -1) if window_size is None else tuple(window_size)
        if block_table is not None and k.ndim == 4:
            k_parts = []
            v_parts = []
            cu_k = [0]
            for i in range(cu_seqlens_k.numel() - 1):
                seq_len = int((cu_seqlens_k[i + 1] - cu_seqlens_k[i]).item())
                k_seq = gather_paged_cache(k, block_table, i, seq_len)
                v_seq = gather_paged_cache(v, block_table, i, seq_len)
                k_parts.append(k_seq)
                v_parts.append(v_seq)
                cu_k.append(cu_k[-1] + k_seq.shape[0])
            k = torch.cat(k_parts, dim=0) if k_parts else k.new_empty((0, self.num_kv_heads, self.head_dim))
            v = torch.cat(v_parts, dim=0) if v_parts else v.new_empty((0, self.num_kv_heads, self.head_dim))
            cu_seqlens_k = torch.tensor(cu_k, device=cu_seqlens_k.device, dtype=cu_seqlens_k.dtype)
        return varlen_attention(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            softmax_scale=kwargs.get("softmax_scale", self.sm_scale),
            causal=kwargs.get("causal", True),
            window_size=window_size,
            s_aux=kwargs.get("s_aux", None),
            softcap=kwargs.get("softcap", 0.0),
        )
