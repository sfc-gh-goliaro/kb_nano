"""Whisper attention variants: encoder self-attention, decoder self-attention,
and decoder cross-attention.

Encoder self-attention has no KV cache and no causal mask (full bidirectional).
Decoder self-attention uses the standard paged KV cache with causal masking.
Cross-attention computes encoder K/V once during prefill and caches them for
subsequent decode steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size
from .parallel_linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from .attention_impl import Attention
from ..L1.flash_attn_prefill import FlashAttnPrefill


class WhisperEncoderSelfAttention(nn.Module):
    """Bidirectional self-attention for the Whisper encoder.

    No KV cache. Uses flash_attn_varlen_func with causal=False on batched
    [B, T, D] inputs.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        tp = _tp_size()
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp
        self.head_dim = embed_dim // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim, self.head_dim,
            num_heads, num_heads,
            bias=True,
        )
        self.out_proj = RowParallelLinear(embed_dim, embed_dim, bias=True)
        self.prefill_op = FlashAttnPrefill(self.num_heads, self.num_heads, self.head_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D] or [T, D]
        Returns:
            [B, T, D] or [T, D]
        """
        is_2d = hidden_states.dim() == 2
        if is_2d:
            hidden_states = hidden_states.unsqueeze(0)

        B, T, D = hidden_states.shape
        flat = hidden_states.reshape(B * T, D)
        qkv = self.qkv_proj(flat)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(B * T, self.num_heads, self.head_dim)
        k = k.view(B * T, self.num_heads, self.head_dim)
        v = v.view(B * T, self.num_heads, self.head_dim)

        cu_seqlens = torch.arange(
            0, (B + 1) * T, T, dtype=torch.int32, device=hidden_states.device,
        )
        out = self.prefill_op(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=T,
            max_seqlen_k=T,
            softmax_scale=self.scale,
            causal=False,
        )

        out = out.reshape(B, T, self.num_heads * self.head_dim)
        out = self.out_proj(out.reshape(B * T, -1))
        out = out.view(B, T, -1)

        if is_2d:
            out = out.squeeze(0)
        return out


class WhisperDecoderSelfAttention(nn.Module):
    """Causal self-attention for the Whisper decoder with paged KV cache.

    Uses the standard Attention implementation from attention_impl.py.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        tp = _tp_size()
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp
        self.head_dim = embed_dim // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim, self.head_dim,
            num_heads, num_heads,
            bias=True,
        )
        self.out_proj = RowParallelLinear(embed_dim, embed_dim, bias=True)
        self.attn = Attention(self.num_heads, self.head_dim, self.scale,
                              num_kv_heads=self.num_heads)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [N, D] flat token embeddings (paged KV cache context)
        """
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        return self.out_proj(attn_output)


class WhisperCrossAttention(nn.Module):
    """Decoder cross-attention: queries from decoder, keys/values from encoder.

    Encoder K/V are projected and cached during the first decode step.
    On subsequent steps, encoder_hidden_states is None and the cached
    K/V are reused.

    Interface matches vLLM's WhisperCrossAttention.forward() signature.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        tp = _tp_size()
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp
        self.head_dim = embed_dim // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=True)
        self.k_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=False)
        self.v_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=True)
        self.out_proj = RowParallelLinear(embed_dim, embed_dim, bias=True)

        self.prefill_op = FlashAttnPrefill(
            self.num_heads, self.num_heads, self.head_dim,
        )

        self._cached_k: torch.Tensor | None = None
        self._cached_v: torch.Tensor | None = None

    def clear_cache(self):
        self._cached_k = None
        self._cached_v = None

    def cache_encoder_kv(self, encoder_hidden_states: torch.Tensor):
        """Pre-compute and cache encoder K/V projections.

        Args:
            encoder_hidden_states: [B, T_enc, D]
        """
        B, T_enc, D = encoder_hidden_states.shape
        enc_flat = encoder_hidden_states.reshape(B * T_enc, D)
        k = self.k_proj(enc_flat)
        v = self.v_proj(enc_flat)
        self._cached_k = k.view(B, T_enc, self.num_heads, self.head_dim)
        self._cached_v = v.view(B, T_enc, self.num_heads, self.head_dim)
        self._cached_T_enc = T_enc

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        num_decoder_seqs: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [N_dec, D] decoder token states (flat)
            encoder_hidden_states: [B, T_enc, D] encoder outputs or None
            num_decoder_seqs: number of decoder sequences in the batch
        """
        if encoder_hidden_states is not None:
            self.cache_encoder_kv(encoder_hidden_states)

        q = self.q_proj(hidden_states)
        N_dec = q.shape[0]

        assert self._cached_k is not None, "Cross-attention cache not initialized"
        cached_k = self._cached_k
        cached_v = self._cached_v
        B = cached_k.shape[0]
        T_enc = self._cached_T_enc

        if num_decoder_seqs is not None:
            B_dec = num_decoder_seqs
        else:
            B_dec = B

        q = q.view(N_dec, self.num_heads, self.head_dim)
        k_flat = cached_k.reshape(B * T_enc, self.num_heads, self.head_dim)
        v_flat = cached_v.reshape(B * T_enc, self.num_heads, self.head_dim)

        tokens_per_dec = N_dec // B_dec
        cu_seqlens_q = torch.arange(
            0, (B_dec + 1) * tokens_per_dec, tokens_per_dec,
            dtype=torch.int32, device=q.device,
        )
        cu_seqlens_k = torch.arange(
            0, (B_dec + 1) * T_enc, T_enc,
            dtype=torch.int32, device=q.device,
        )

        out = self.prefill_op(
            q, k_flat, v_flat,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=tokens_per_dec,
            max_seqlen_k=T_enc,
            softmax_scale=self.scale,
            causal=False,
        )

        out = out.view(N_dec, self.num_heads * self.head_dim)
        return self.out_proj(out)
