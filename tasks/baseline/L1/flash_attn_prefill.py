"""Flash attention prefill kernel (variable-length sequences)."""

import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func


class FlashAttnPrefill(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        if torch.compiler.is_compiling():
            softmax_scale = kwargs.get("softmax_scale", self.sm_scale)
            return torch.ops.kb_nano.flash_attn_prefill(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, softmax_scale,
            )
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs,
        )
