"""Semantic PyTorch reference for flash_attn_decode.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kb_nano.tasks.reference.L1._attention import dense_attention, gather_paged_cache


class FlashAttnDecode(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(self, q, k_cache, v_cache, cache_seqlens=None, **kwargs):
        block_table = kwargs.get("block_table", None)
        softmax_scale = kwargs.get("softmax_scale", self.head_dim ** -0.5)
        window_size = kwargs.get("window_size", (-1, -1))
        window_size = (-1, -1) if window_size is None else tuple(window_size)
        s_aux = kwargs.get("s_aux", None)
        softcap = kwargs.get("softcap", 0.0)
        if cache_seqlens is None:
            cache_seqlens = torch.full((q.shape[0],), k_cache.shape[0], device=q.device, dtype=torch.int32)

        outs = []
        for i in range(q.shape[0]):
            seq_len = int(cache_seqlens[i].item())
            k = gather_paged_cache(k_cache, block_table, i, seq_len)
            v = gather_paged_cache(v_cache, block_table, i, seq_len)
            out = dense_attention(
                q[i:i + 1].unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                softmax_scale=softmax_scale, causal=True,
                window_size=window_size, s_aux=s_aux, softcap=softcap,
            ).squeeze(0).squeeze(0)
            outs.append(out)
        return torch.stack(outs, dim=0)
