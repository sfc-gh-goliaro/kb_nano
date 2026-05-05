"""Jamba's multi-head attention block (no RoPE, no QK-norm).

Reference: ``transformers.models.jamba.modeling_jamba.JambaAttention``.

Key difference from :class:`L2.attention.LlamaAttention`: Jamba does NOT
use rotary position embeddings.  Position information enters the network
through Mamba's selective scan (data-dependent recurrence) plus the
ordering of attention layers; no positional embedding is applied to Q
or K.

Layout: works with HuggingFace-style batched ``[B, T, hidden]`` input,
emits ``[B, T, hidden]`` output.  KV cache is owned by the engine
(``JambaCacheView``) and passed as ``past_key`` / ``past_value`` slabs
that we concatenate to the new K / V before the attention call.

L1 ops: ``Linear`` (q/k/v/o projections), ``DenseAttention`` (kernel
launch).  No ``F.linear``/``F.scaled_dot_product_attention`` leak.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear


class JambaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = Linear(
            hidden_size, num_attention_heads * self.head_dim, bias=False,
        )
        self.k_proj = Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=False,
        )
        self.v_proj = Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=False,
        )
        self.o_proj = Linear(
            num_attention_heads * self.head_dim, hidden_size, bias=False,
        )

        # cuDNN backend gives the best B200 perf for this layout
        # (see CLAUDE.md: ~2.7x over cutlass FMHA).  ``DenseAttention``
        # falls back to MATH for masks cuDNN can't handle.
        self.attn = DenseAttention(backend="cudnn")

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """[B, H_kv, T, D] -> [B, H_kv * n_rep, T, D]. No-op if n_rep==1."""
        if n_rep == 1:
            return x
        b, h, t, d = x.shape
        return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_writeback: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        hidden_states : [B, T, hidden]
        past_key / past_value : [B, H_kv, S, D] (S = number of cached
            tokens) or None for the first call.
        attention_mask : optional additive mask of shape ``[B, 1, T, T+S]``
            (broadcast over heads).  Standard HF semantics: 0 for valid
            positions, ``-inf`` (or large negative) for masked.  Caller
            handles causality there.
        cache_writeback : optional ``(k_buf, v_buf)`` pre-allocated
            cache slabs.  When provided, the new (concatenated) K/V are
            copied into ``k_buf[:, :, :T+S, :]`` so the caller can hold
            on to a single growing tensor without per-step alloc.
        """
        b, t, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(b, t, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim)

        # Move to [B, H, T, D] for concat with the cache.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_key is not None and past_value is not None:
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        if cache_writeback is not None:
            k_buf, v_buf = cache_writeback
            s = k.size(2)
            k_buf[:, :, :s, :].copy_(k)
            v_buf[:, :, :s, :].copy_(v)

        k_full = self._repeat_kv(k, self.num_kv_groups)
        v_full = self._repeat_kv(v, self.num_kv_groups)

        # ``DenseAttention`` consumes [B, T, H, D].
        q_ = q.transpose(1, 2).contiguous()
        k_ = k_full.transpose(1, 2).contiguous()
        v_ = v_full.transpose(1, 2).contiguous()

        # Causal masking: when no mask is given, rely on ``causal=True``
        # ONLY for the first call (T == S+T, i.e. T_q == T_kv).  For
        # decode (T_q=1, T_kv>1) ``causal=True`` would shift the
        # diagonal incorrectly, so callers are expected to pass either
        # a proper mask or rely on T_q==1 + the trivial diagonal.
        if attention_mask is None:
            # T_q == T_kv => causal=True is correct.
            # T_q < T_kv  => decode step, all KV positions are visible.
            causal = (q_.size(1) == k_.size(1))
            out = self.attn(q_, k_, v_, softmax_scale=self.scaling, causal=causal)
        else:
            out = self.attn(q_, k_, v_, softmax_scale=self.scaling,
                            attn_mask=attention_mask)

        # [B, T, H, D] -> [B, T, hidden]
        out = out.contiguous().view(b, t, self.num_heads * self.head_dim)
        return self.o_proj(out), k, v
