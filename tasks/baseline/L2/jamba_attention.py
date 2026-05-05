"""Jamba's multi-head attention block (no RoPE, no QK-norm).

Reference: ``transformers.models.jamba.modeling_jamba.JambaAttention``.

Key difference from :class:`L2.attention.LlamaAttention`: Jamba does NOT
use rotary position embeddings.  Position information enters the network
through Mamba's selective scan (data-dependent recurrence) plus the
ordering of attention layers; no positional embedding is applied to Q
or K.

Layout: works with HuggingFace-style batched ``[B, T, hidden]`` input,
emits ``[B, T, hidden]`` output.  KV cache is owned by the engine and
read from the global ``Context`` (populated via ``set_jamba_context``),
matching the project's ``set_context`` / ``get_context()`` convention
used by Llama / Mamba / Mamba2 / Mixtral.

L1 ops: ``Linear`` (q/k/v/o projections), ``DenseAttention`` (kernel
launch).  No ``F.linear``/``F.scaled_dot_product_attention`` leak.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.context import get_context
from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear


class JambaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int | None = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.layer_idx = layer_idx

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
        positions: torch.Tensor | None,    # unused (no RoPE); kept for
                                            # signature parity with Llama.
        hidden_states: torch.Tensor,        # [B, T, hidden]
    ) -> torch.Tensor:
        """Forward.  Reads per-step KV state and attention mask from the
        global ``Context`` populated by the engine's ``set_jamba_context``.

        Two modes are dispatched off ``ctx.jamba_attn_metadata``:

          * **Static-shape decode** (CUDA-graph friendly).  ``meta.kv_slabs``
            is a list of ``(k_buf, v_buf)`` of shape
            ``[B, H_kv, max_total, D]``.  The new K/V are written via
            ``index_copy_`` at ``meta.slot_pos`` and attention is
            computed against the **full** slab; ``meta.attn_mask_4d``
            masks out future and pad positions.

          * **Eager (variable shape)** prefill / decode.  ``meta.past_kv``
            holds per-layer ``(past_k, past_v)`` slabs that we
            concatenate to the new tokens; ``meta.cache_writeback``
            optionally provides a pre-allocated ``(k_buf, v_buf)`` to
            copy the concat result into.
        """
        ctx = get_context()
        meta = ctx.jamba_attn_metadata
        assert meta is not None, (
            "JambaAttention.forward called without a JambaAttnMetadata "
            "installed on the global Context (use set_jamba_context)."
        )

        b, t, _ = hidden_states.shape

        # ------------------------------------------------------------------
        # Static-shape decode path (CUDA-graph friendly).
        # ------------------------------------------------------------------
        if meta.kv_slabs is not None and t == 1:
            k_slab, v_slab = meta.kv_slabs[self.layer_idx]
            slot_pos = meta.slot_pos
            attention_mask = meta.attn_mask_4d

            q = self.q_proj(hidden_states).view(b, 1, self.num_heads, self.head_dim)
            k = self.k_proj(hidden_states).view(b, 1, self.num_kv_heads, self.head_dim)
            v = self.v_proj(hidden_states).view(b, 1, self.num_kv_heads, self.head_dim)

            # Move to [B, H_kv, 1, D] before writing into the slab.
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # In-place write at slot_pos.  ``index_copy_`` accepts a 1-d
            # index tensor, so we view the 0-d scalar as 1-d.  This is
            # CUDA-graph safe: the index tensor's storage is fixed; the
            # caller mutates its **value** in-place between replays.
            slot_idx_1d = slot_pos.view(1)
            k_slab.index_copy_(2, slot_idx_1d, k)
            v_slab.index_copy_(2, slot_idx_1d, v)

            # Attention against the full slab.  The caller-supplied mask
            # is what enforces "ignore positions >= cur_len" and the
            # original left-padding -- it is shape [B, 1, 1, max_total].
            k_full = self._repeat_kv(k_slab, self.num_kv_groups)
            v_full = self._repeat_kv(v_slab, self.num_kv_groups)

            # DenseAttention wants [B, T, H, D].
            q_ = q.contiguous()
            k_ = k_full.transpose(1, 2).contiguous()
            v_ = v_full.transpose(1, 2).contiguous()
            out = self.attn(
                q_, k_, v_,
                softmax_scale=self.scaling, attn_mask=attention_mask,
            )
            out = out.contiguous().view(b, 1, self.num_heads * self.head_dim)
            return self.o_proj(out)

        # ------------------------------------------------------------------
        # Eager (variable-shape) path -- prefill or non-graph decode.
        # ------------------------------------------------------------------
        past_key = past_value = None
        if meta.past_kv is not None:
            pkv = meta.past_kv[self.layer_idx]
            if pkv is not None:
                past_key, past_value = pkv

        cache_writeback = None
        if meta.cache_writeback is not None:
            cache_writeback = meta.cache_writeback[self.layer_idx]

        attention_mask = meta.attn_mask_4d

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
            causal = (q_.size(1) == k_.size(1))
            out = self.attn(q_, k_, v_, softmax_scale=self.scaling, causal=causal)
        else:
            out = self.attn(q_, k_, v_, softmax_scale=self.scaling,
                            attn_mask=attention_mask)

        # [B, T, H, D] -> [B, T, hidden]
        out = out.contiguous().view(b, t, self.num_heads * self.head_dim)
        return self.o_proj(out)
