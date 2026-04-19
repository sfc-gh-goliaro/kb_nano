"""Dense (non-paged) multi-head attention.

Unlike the paged attention ops (FlashAttnPrefill/Decode) which use KV cache
and varlen APIs, this op handles full dense attention with a standard
(batch, seq_len, num_heads, head_dim) layout. Supports both causal and
non-causal modes.

Backend selection is controlled via the ``backend`` parameter:

  ``"auto"`` (default) — picks the fastest available backend:
    Ampere / Hopper (cc 8.x–9.x):
      FA3 via ``fa3_fwd_interface`` > ``flash_attn_interface`` > FA2 via
      ``flash_attn`` > PyTorch SDPA.
    Blackwell+ (cc >= 10.0) or pre-Ampere:
      PyTorch SDPA (dispatches to cuDNN flash attention on supported GPUs).

  ``"sdpa"`` — always use ``F.scaled_dot_product_attention``.  Fully
    ``torch.compile``-friendly and produces numerically identical results
    to diffusers' ``AttnProcessor2_0``.

  ``"flash_attn"`` — always use the flash-attention fallback chain
    (FA3 > FA2); raises if none is installed.

Used by diffusion models (FLUX, SDXL) and any architecture that needs
stateless multi-head attention without KV cache.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_flash_attn_func():
    """Return the best available flash-attention callable, or None."""
    for import_path in (
        ("fa3_fwd_interface", "flash_attn_func"),
        ("flash_attn_interface", "flash_attn_func"),
        ("flash_attn", "flash_attn_func"),
    ):
        try:
            mod = __import__(import_path[0], fromlist=[import_path[1]])
            return getattr(mod, import_path[1])
        except ImportError:
            continue
    return None


class DenseAttention(nn.Module):
    """Dense multi-head attention.

    Input layout: (batch, seq_len, num_heads, head_dim).

    Args:
        backend: Which kernel to use.
            ``"auto"`` selects flash-attention on Ampere/Hopper when
            available, SDPA everywhere else.
            ``"sdpa"`` always uses ``F.scaled_dot_product_attention``.
            ``"flash_attn"`` always uses the flash-attention package.
    """

    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn"] = "auto"):
        super().__init__()
        self.fa_func = None

        if backend == "sdpa":
            return

        if backend == "flash_attn":
            self.fa_func = _resolve_flash_attn_func()
            if self.fa_func is None:
                raise ImportError(
                    "backend='flash_attn' requested but no flash-attention "
                    "package is installed (tried fa3_fwd_interface, "
                    "flash_attn_interface, flash_attn)"
                )
            return

        # backend == "auto"
        cc = (torch.cuda.get_device_capability()[0] * 10
              + torch.cuda.get_device_capability()[1])
        if 80 <= cc < 100:
            self.fa_func = _resolve_flash_attn_func()

    def forward(self, query, key, value, softmax_scale=None, causal=False):
        if self.fa_func is not None and query.dtype != torch.float32:
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
