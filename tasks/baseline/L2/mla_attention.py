"""MLA (Multi-head Latent Attention) for Kimi-Linear (L2).

DeepSeek-V2 style compressed KV attention (NO RoPE rotation):
  q = q_proj(x)                     -> split to (q_nope, q_rope)
  kv_a = kv_a_proj_with_mqa(x)      -> split to (kv_compressed, k_rope_shared)
  kv_compressed = kv_a_layernorm(kv_compressed)
  kv_b = kv_b_proj(kv_compressed)   -> split to (k_nope, v)
  k = cat(k_nope, k_rope)           (no rotation applied)
  q = cat(q_nope, q_rope)           (no rotation applied)
  attn = SDPA(q, k, v)              FA3 varlen on paged KV cache
  output = o_proj(attn)

Operates on a flat varlen batch ``[num_actual_tokens, hidden_size]`` with
per-request metadata supplied via ``KimiLinearMetadata``. Stores K/V into
the per-layer paged-KV cache owned by ``KimiLinearStateManager`` using
``slot_mapping``; runs FA3 (asymmetric ``head_dim_qk != head_dim_v`` is
supported on Hopper / SM90+) over the full per-sequence cached context.

Composes only L1 ops (``RMSNorm``, ``StoreKVCache``, ``FlashAttnPrefill``,
``FlashAttnDecode``) and the canonical TP linears in ``parallel_linear``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.kimi_linear_metadata import get_metadata
from ....infra.tp import _tp_size
from ..L1.flash_attn_decode import FlashAttnDecode
from ..L1.flash_attn_prefill import FlashAttnPrefill
from ..L1.rms_norm import RMSNorm
from ..L1.store_kvcache import StoreKVCache
from .parallel_linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


class MLAAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2 style) with paged KV."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        layer_idx: int,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        tp = _tp_size()
        self.layer_idx = layer_idx
        self.num_heads = num_attention_heads
        self.local_num_heads = num_attention_heads // tp
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.scaling = self.qk_head_dim ** -0.5

        self.q_proj = ColumnParallelLinear(
            hidden_size, num_attention_heads * self.qk_head_dim
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False,
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank, num_attention_heads * (qk_nope_head_dim + v_head_dim)
        )
        self.o_proj = RowParallelLinear(
            num_attention_heads * v_head_dim, hidden_size
        )

        self.store_kvcache = StoreKVCache()
        self.flash_attn_prefill = FlashAttnPrefill(
            self.local_num_heads, self.local_num_heads, self.qk_head_dim,
        )
        self.flash_attn_decode = FlashAttnDecode(
            self.local_num_heads, self.local_num_heads, self.qk_head_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_manager=None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [num_actual_tokens, hidden_size] flat varlen batch
            state_manager: ``KimiLinearStateManager`` (paged KV owner).
        Returns:
            output: [num_actual_tokens, hidden_size]
        """
        md = get_metadata()
        N = hidden_states.shape[0]

        if md is None or state_manager is None:
            return torch.zeros_like(hidden_states)

        layer_idx = self.layer_idx

        q = self.q_proj(hidden_states)
        q = q.view(N, self.local_num_heads, self.qk_head_dim)

        kv_a = self.kv_a_proj_with_mqa(hidden_states)
        kv_compressed = kv_a[..., :self.kv_lora_rank]
        k_rope_shared = kv_a[..., self.kv_lora_rank:]

        kv_compressed = self.kv_a_layernorm(kv_compressed)

        kv_b = self.kv_b_proj(kv_compressed)
        kv_b = kv_b.view(
            N, self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope = kv_b[..., :self.qk_nope_head_dim]
        v_unpadded = kv_b[..., self.qk_nope_head_dim:]

        k_rope = k_rope_shared.unsqueeze(1).expand(
            N, self.local_num_heads, self.qk_rope_head_dim,
        )
        k = torch.cat([k_nope, k_rope], dim=-1).contiguous()

        # Pad V to qk_head_dim so K and V share head dim -> can use the
        # standard FA3 varlen kernel (which requires headdim_q == headdim_kv).
        # The trailing ``qk_rope_head_dim`` slots are zeros; we slice them
        # back off the attention output before ``o_proj``. Matches vLLM's
        # non-absorbed MLA path.
        v_padded = torch.nn.functional.pad(
            v_unpadded, (0, self.qk_head_dim - self.v_head_dim),
        ).contiguous()

        k_cache = state_manager.k_cache[layer_idx]
        v_cache = state_manager.v_cache[layer_idx]
        # store_kvcache reads ``key.shape`` to size the per-token stride,
        # so we pass V with the padded layout that matches v_cache's shape.
        self.store_kvcache(k, v_padded, k_cache, v_cache, md.slot_mapping)

        out = torch.empty(
            N, self.local_num_heads, self.qk_head_dim,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )

        nd = md.num_decodes
        ndt = md.num_decode_tokens
        np_ = md.num_prefills
        npt = md.num_prefill_tokens

        # FA3 requires int32 cu_seqlens / seqused_k.
        if nd > 0:
            cache_seqlens = md.seq_lens[:nd].to(torch.int32)
            decode_block_tables = md.block_tables[:nd]
            out[:ndt] = self.flash_attn_decode(
                q[:ndt], k_cache, v_cache,
                cache_seqlens=cache_seqlens,
                block_table=decode_block_tables,
                softmax_scale=self.scaling,
                causal=True,
                max_seq_len=md.max_seq_len,
            )

        if np_ > 0:
            cu_pf = (md.query_start_loc[nd:] - md.query_start_loc[nd]).to(torch.int32)
            seqs_k = md.seq_lens[nd:]
            cu_k_pf = torch.zeros(np_ + 1, dtype=torch.int32, device=q.device)
            cu_k_pf[1:] = torch.cumsum(seqs_k.to(torch.int32), dim=0)
            prefill_block_tables = md.block_tables[nd:]
            out[ndt:] = self.flash_attn_prefill(
                q[ndt:], k_cache, v_cache,
                cu_seqlens_q=cu_pf,
                cu_seqlens_k=cu_k_pf,
                max_seqlen_q=md.max_query_len,
                max_seqlen_k=md.max_seq_len,
                softmax_scale=self.scaling,
                causal=True,
                block_table=prefill_block_tables,
            )

        # Slice the v_head_dim suffix back off (it was zero-padded for FA).
        out_v = out[..., : self.v_head_dim].contiguous()
        return self.o_proj(out_v.reshape(N, self.local_num_heads * self.v_head_dim))
