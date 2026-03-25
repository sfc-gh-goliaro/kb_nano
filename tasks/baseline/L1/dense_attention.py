"""Dense (non-paged) multi-head attention.

Unlike the paged attention ops (FlashAttnPrefill/Decode) which use KV cache
and varlen APIs, this op handles full dense attention with a standard
(batch, seq_len, num_heads, head_dim) layout. Supports both causal and
non-causal modes.

Backend selection (Ampere / Hopper, cc 8.x–9.x):
  1. FA3 via ``fa3_fwd_interface`` (fa3-fwd PyPI package)
  2. FA3 via ``flash_attn_interface`` (source-built flash-attention)
  3. FA2 via ``flash_attn`` (flash-attn PyPI package)
  This mirrors the fallback chain used by vllm-omni.

Backend selection (Blackwell+, cc >= 10.0):
  PyTorch SDPA, which dispatches to cuDNN flash attention — significantly
  faster than the flash-attn package on these GPUs.

Used by diffusion models (FLUX) and any architecture that needs stateless
multi-head attention without KV cache.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseAttention(nn.Module):
    """Dense multi-head attention.

    Input layout: (batch, seq_len, num_heads, head_dim).
    """

    def __init__(self):
        super().__init__()
        self.fa_func = None

        cc = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]
        if cc < 80 or cc >= 100:
            return

        try:
            from fa3_fwd_interface import flash_attn_func
            self.fa_func = flash_attn_func
        except ImportError:
            pass

        if self.fa_func is None:
            try:
                from flash_attn_interface import flash_attn_func
                self.fa_func = flash_attn_func
            except ImportError:
                pass

        if self.fa_func is None:
            try:
                from flash_attn import flash_attn_func
                self.fa_func = flash_attn_func
            except ImportError:
                pass

    def forward(self, query, key, value, softmax_scale=None, causal=False):
        if self.fa_func is not None:
            out = self.fa_func(
                query, key, value,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out

        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        return out.permute(0, 2, 1, 3)
