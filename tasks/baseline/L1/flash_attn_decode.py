"""Flash attention decode kernel (with paged KV cache).

Accepts 3D ``[N, H, D]`` query input (matching TRTLLMDecode's interface)
and handles the unsqueeze/squeeze internally for flash_attn_with_kvcache
which expects ``[N, 1, H, D]``.
"""

import torch
import torch.nn as nn
from flash_attn import flash_attn_with_kvcache


class FlashAttnDecode(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(self, q, k_cache, v_cache, cache_seqlens=None, **kwargs):
        if torch.compiler.is_compiling():
            max_seq_len = kwargs.get("max_seq_len", 0)
            block_table = kwargs.get("block_table", None)
            softmax_scale = kwargs.get("softmax_scale", self.head_dim ** -0.5)
            return torch.ops.kb_nano.flash_attn_decode(
                q, k_cache, v_cache, cache_seqlens,
                block_table, softmax_scale, max_seq_len,
            )
        kwargs.pop("max_seq_len", None)
        return flash_attn_with_kvcache(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=cache_seqlens, **kwargs,
        ).squeeze(1)
