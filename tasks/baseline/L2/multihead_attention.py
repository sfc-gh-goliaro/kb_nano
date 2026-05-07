"""MultiheadAttention L2 wrapper.

Mirrors ``torch.nn.MultiheadAttention``'s public interface but uses kb-nano's
``DenseAttention`` (SDPA) when ``need_weights=False`` — which is the common
case in HF (e.g. ``aria``, ``idefics2`` use ``self.attention(q,k,v)[0]``,
discarding weights). When ``need_weights=True`` the wrapper falls back to
explicit matmul + softmax + matmul so the attention map is materialized.

Why an L2 wrapper rather than ``nn.MultiheadAttention`` directly:
  - ``nn.MultiheadAttention`` always materializes the attention map, which
    defeats the SDPA fast path.
  - The ``in_proj_weight`` / ``in_proj_bias`` packed-Q-K-V layout matches
    HF reference checkpoints exactly so ``state_dict`` loads with no remap.
  - The output shape and the optional weight tuple match
    ``nn.MultiheadAttention``'s signature so call sites in ``aria``,
    ``bridgetower``, ``idefics2``, ``mask2former``, ``oneformer``,
    ``omdet_turbo``, ``phi4_multimodal`` etc. are drop-in.

Per-head dim must satisfy ``embed_dim % num_heads == 0``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.dense_attention import DenseAttention


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
    ):
        super().__init__()
        if add_bias_kv or add_zero_attn:
            raise NotImplementedError("add_bias_kv / add_zero_attn not supported by this wrapper")
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.dropout = dropout
        self.batch_first = batch_first

        # Match torch.nn.MultiheadAttention's parameter naming so HF checkpoints load.
        if self._qkv_same_embed_dim:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)
        else:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim)))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim)))
            self.register_parameter("in_proj_weight", None)
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._sdpa = DenseAttention()

    def _in_proj(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """Return (q, k, v) projections matching torch.nn.MultiheadAttention's
        packed-weight semantics."""
        if self._qkv_same_embed_dim and (query is key) and (key is value):
            # Self-attention fast path: single matmul on the combined weight.
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
        elif self._qkv_same_embed_dim:
            E = self.embed_dim
            w_q, w_k, w_v = self.in_proj_weight[:E], self.in_proj_weight[E:2 * E], self.in_proj_weight[2 * E:]
            if self.in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = self.in_proj_bias[:E], self.in_proj_bias[E:2 * E], self.in_proj_bias[2 * E:]
            q = F.linear(query, w_q, b_q)
            k = F.linear(key, w_k, b_k)
            v = F.linear(value, w_v, b_v)
        else:
            if self.in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                E = self.embed_dim
                b_q, b_k, b_v = self.in_proj_bias[:E], self.in_proj_bias[E:2 * E], self.in_proj_bias[2 * E:]
            q = F.linear(query, self.q_proj_weight, b_q)
            k = F.linear(key, self.k_proj_weight, b_k)
            v = F.linear(value, self.v_proj_weight, b_v)
        return q, k, v

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # nn.MultiheadAttention default convention: (L, B, E) unless batch_first.
        if self.batch_first:
            # (B, L, E) — work in (B, L, E) throughout
            B, Lq, _ = query.shape
            Lk = key.shape[1]
        else:
            # (L, B, E) — transpose to (B, L, E) for our internal layout
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            B, Lq, _ = query.shape
            Lk = key.shape[1]

        q, k, v = self._in_proj(query, key, value)

        # reshape to multi-head
        q = q.view(B, Lq, self.num_heads, self.head_dim)
        k = k.view(B, Lk, self.num_heads, self.head_dim)
        v = v.view(B, Lk, self.num_heads, self.head_dim)

        # Build the combined attention mask: merge attn_mask + key_padding_mask
        # into a single additive [B, num_heads, Lq, Lk] tensor with -inf at masked positions.
        merged_mask = self._merge_masks(attn_mask, key_padding_mask, B, Lq, Lk, q.dtype, q.device)

        # SDPA fast path: convert to (B, H, L, D) for F.scaled_dot_product_attention.
        # Don't materialize attention weights when need_weights=False.
        if not need_weights:
            q_s = q.transpose(1, 2)
            k_s = k.transpose(1, 2)
            v_s = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q_s, k_s, v_s,
                attn_mask=merged_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            out = out.transpose(1, 2).reshape(B, Lq, self.embed_dim)
            attn_out = self.out_proj(out)
            if not self.batch_first:
                attn_out = attn_out.transpose(0, 1)
            return attn_out, None

        # need_weights=True: explicit attention map (matches nn.MultiheadAttention's behavior)
        scale = 1.0 / math.sqrt(self.head_dim)
        # (B, H, Lq, D) x (B, H, D, Lk) -> (B, H, Lq, Lk)
        attn = torch.matmul(
            (q * scale).transpose(1, 2),
            k.transpose(1, 2).transpose(-2, -1),
        )
        if merged_mask is not None:
            attn = attn + merged_mask
        if is_causal:
            causal = torch.triu(
                torch.full((Lq, Lk), float("-inf"), device=attn.device, dtype=attn.dtype),
                diagonal=1,
            )
            attn = attn + causal
        attn_w = F.softmax(attn, dim=-1)
        if self.training and self.dropout > 0:
            attn_w = F.dropout(attn_w, p=self.dropout)
        # (B, H, Lq, Lk) x (B, H, Lk, D) -> (B, H, Lq, D)
        out = torch.matmul(attn_w, v.transpose(1, 2))
        out = out.transpose(1, 2).reshape(B, Lq, self.embed_dim)
        attn_out = self.out_proj(out)
        if not self.batch_first:
            attn_out = attn_out.transpose(0, 1)
        # Return per-head or averaged weights to match nn.MultiheadAttention semantics
        if average_attn_weights:
            return attn_out, attn_w.mean(dim=1)  # (B, Lq, Lk)
        return attn_out, attn_w  # (B, H, Lq, Lk)

    @staticmethod
    def _merge_masks(
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        B: int, Lq: int, Lk: int,
        dtype: torch.dtype, device: torch.device,
    ) -> torch.Tensor | None:
        """Merge attn_mask + key_padding_mask into a single additive mask.
        Output shape is broadcastable to [B, num_heads, Lq, Lk]; None means no mask."""
        out: torch.Tensor | None = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                am = torch.zeros_like(attn_mask, dtype=dtype, device=device)
                am.masked_fill_(attn_mask, float("-inf"))
            else:
                am = attn_mask.to(dtype).to(device)
            # Reshape to [1, 1, Lq, Lk] or accept [Lq, Lk]
            if am.dim() == 2:
                am = am.unsqueeze(0).unsqueeze(0)
            elif am.dim() == 3:
                am = am.unsqueeze(1)  # [B, 1, Lq, Lk]
            out = am
        if key_padding_mask is not None:
            # Standard convention: True = pad (mask out)
            kpm = key_padding_mask.to(device)
            if kpm.dtype == torch.bool:
                m = torch.zeros((B, 1, 1, Lk), dtype=dtype, device=device)
                m.masked_fill_(kpm.view(B, 1, 1, Lk), float("-inf"))
            else:
                m = kpm.to(dtype).view(B, 1, 1, Lk)
            out = m if out is None else (out + m)
        return out
