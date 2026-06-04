"""DeepSeek V4 MLA attention (model-level).

V4 attention differs from V3:
- head_dim=512 directly stored in KV cache (no kv_lora_rank compression)
- nope_head_dim = head_dim - qk_rope_head_dim = 448
- KV is stored directly (no kv_b_proj, no weight absorption)
- Output uses grouped wo_a + wo_b
- Per-head RMSNorm on Q (no learnable weights)
- attn_sink per-head learnable parameter
- v_head_dim = head_dim = 512 (value = full KV vector)

Cache stores head_dim=512 BF16 dims per token (nope_448 + rope_64_with_RoPE).
Decode uses FlashMLA with head_dim_v=512.

Reference: vllm/model_executor/layers/deepseek_v4_attention.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size
from ....infra.context import get_context
from .parallel_linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear,
)
from ..L1.rms_norm import RMSNorm
from ..L1.flash_mla_decode import (
    FlashMLADecode,
    FlashMLAGetMetadata,
)
from ..L1.merge_attn_states import MergeAttnStates

_MLA_HEAD_DIM_V = 512
_CACHE_DIM = 512


class DeepSeekV4Attention(nn.Module):
    """DeepSeek V4 Multi-head Latent Attention.

    Forward: fused_wqa_wkv -> norms -> wq_b -> per-head norm -> RoPE
             -> store KV cache -> flash attention/MLA decode -> wo_a -> wo_b
    """

    def __init__(self, config, rotary_emb: nn.Module,
                 quant_config: dict | None = None,
                 compress_ratio: int = 1,
                 topk_indices_buffer: torch.Tensor | None = None):
        super().__init__()
        tp = _tp_size()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.num_heads = config.num_attention_heads
        self.num_local_heads = self.num_heads // tp
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp
        self.compress_ratio = compress_ratio
        self.sliding_window = config.sliding_window

        self.softmax_scale = self.head_dim ** -0.5
        self.rotary_emb = rotary_emb

        # KV cache attrs for engine discovery (same interface as MLAAttention)
        self.kv_lora_rank = self.nope_head_dim  # 448
        self._num_kv_heads = 1
        self._head_dim = _CACHE_DIM
        self.k_cache = self.v_cache = torch.tensor([])
        self.kv_cache_dtype = "auto"

        # Fused Q + KV projection: [hidden_size -> q_lora_rank + head_dim]
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            quant_config=quant_config,
            disable_tp=True,
        )

        # Q path
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            quant_config=quant_config,
        )

        # KV norm
        self.kv_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Output projection: wo_a (grouped via ColumnParallel) + wo_b
        # wo_a is TP-sharded on output: [n_groups*o_lora_rank, input_per_group]
        # -> [n_local_groups*o_lora_rank, heads_per_group*head_dim] per rank
        # Applied as grouped BMM: einsum("bhr,hdr->bhd")
        self.heads_per_group = self.num_local_heads // self.n_local_groups
        wo_a_in = self.num_heads * self.head_dim // self.n_groups  # per-group input
        self.wo_a = ColumnParallelLinear(
            wo_a_in,
            self.n_groups * self.o_lora_rank,
            quant_config=quant_config,
        )
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            quant_config=quant_config,
        )

        # attn_sink: per-head learnable scalar
        padded_heads = max(self.num_local_heads, 64)
        self.attn_sink = nn.Parameter(
            torch.full((padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        # Cached BF16 dequant of wo_a (lazily populated on first forward)
        self._wo_a_bf16: torch.Tensor | None = None

        # Kernel wrappers
        self.decode_op = FlashMLADecode()
        self.get_metadata = FlashMLAGetMetadata()
        self.merge_states = MergeAttnStates()

    def _store_kv(self, kv_full: torch.Tensor, kv_cache: torch.Tensor,
                  slot_mapping: torch.Tensor):
        """Store V4 KV into paged cache. kv_full has RoPE already applied."""
        block_size = kv_cache.shape[1]
        block_idx = slot_mapping // block_size
        slot_idx = slot_mapping % block_size
        kv_cache[block_idx, slot_idx] = kv_full.to(kv_cache.dtype)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        N = hidden_states.shape[0]

        # Fused Q + KV projection
        qkv = self.fused_wqa_wkv(hidden_states)
        qr, kv = qkv.split([self.q_lora_rank, self.head_dim], dim=-1)

        # Q path: norm -> project -> reshape -> per-head RMSNorm
        qr = self.q_norm(qr)
        q = self.wq_b(qr)
        q = q.view(N, self.num_local_heads, self.head_dim)

        q_rms = torch.rsqrt(q.float().square().mean(-1, keepdim=True) + 1e-6)
        q = (q.float() * q_rms).to(q.dtype)

        # KV path: norm then split nope/rope and apply RoPE
        kv = self.kv_norm(kv)
        kv_nope = kv[:, :self.nope_head_dim]
        k_pe = kv[:, self.nope_head_dim:]

        q_rope = q[..., self.nope_head_dim:]
        q[..., self.nope_head_dim:], k_pe_r = self.rotary_emb(
            positions, q_rope, k_pe.unsqueeze(1),
        )
        k_pe_r = k_pe_r.squeeze(1)

        kv_full = torch.cat([kv_nope, k_pe_r], dim=-1)

        # Store in KV cache
        ctx = get_context()
        kv_cache = self.k_cache
        if kv_cache.numel() and ctx.slot_mapping is not None:
            self._store_kv(kv_full, kv_cache, ctx.slot_mapping)

        # Attention dispatch
        if ctx.is_mixed:
            o = self._forward_mixed(q, kv_full, kv_cache, ctx)
        elif ctx.is_prefill:
            o = self._forward_prefill(q, kv_full, ctx)
        else:
            o = self._forward_decode(q, kv_cache, ctx)

        # o: (N, num_local_heads, head_dim) -> grouped output projection
        # Reshape: (N, n_local_groups, heads_per_group * head_dim)
        o_grouped = o.view(N, self.n_local_groups,
                           self.heads_per_group * self.head_dim)

        wa = self._get_wo_a_bf16().view(
            self.n_local_groups, self.o_lora_rank, -1,
        )
        z = torch.einsum("bgr,gdr->bgd", o_grouped.to(wa.dtype), wa)
        return self.wo_b(z.reshape(N, -1).to(o.dtype))

    def _get_wo_a_bf16(self) -> torch.Tensor:
        """Return wo_a weight in BF16, dequantizing from FP8 if needed."""
        if not self.wo_a.use_fp8:
            return self.wo_a.weight.data
        if self._wo_a_bf16 is not None:
            return self._wo_a_bf16
        w_fp8 = self.wo_a.weight.data  # (out, in) fp8
        s = self.wo_a.weight_scale_inv.data  # (ceil(out/128), ceil(in/128)) fp32
        out_dim, in_dim = w_fp8.shape
        bs = 128
        w_bf16 = torch.empty_like(w_fp8, dtype=torch.bfloat16)
        for i in range(0, out_dim, bs):
            for j in range(0, in_dim, bs):
                si, sj = i // bs, j // bs
                block = w_fp8[i:i+bs, j:j+bs].float() * s[si, sj]
                w_bf16[i:i+bs, j:j+bs] = block.to(torch.bfloat16)
        self._wo_a_bf16 = w_bf16
        return w_bf16

    def _forward_prefill(self, q, kv_full, ctx):
        """Dense prefill using PyTorch SDPA (supports head_dim=512)."""
        N = q.shape[0]
        cu_seqlens = ctx.cu_seqlens_q
        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, N], dtype=torch.int32, device=q.device)

        k_exp = kv_full.unsqueeze(1).expand(-1, self.num_local_heads, -1)
        v_exp = k_exp

        num_seqs = cu_seqlens.shape[0] - 1
        if num_seqs == 1:
            q_t = q.unsqueeze(0).transpose(1, 2)
            k_t = k_exp.unsqueeze(0).transpose(1, 2)
            v_t = v_exp.unsqueeze(0).transpose(1, 2)
            o = torch.nn.functional.scaled_dot_product_attention(
                q_t, k_t, v_t, scale=self.softmax_scale, is_causal=True,
            )
            return o.transpose(1, 2).squeeze(0)

        outputs = []
        for s in range(num_seqs):
            start = cu_seqlens[s].item()
            end = cu_seqlens[s + 1].item()
            q_t = q[start:end].unsqueeze(0).transpose(1, 2)
            k_t = k_exp[start:end].unsqueeze(0).transpose(1, 2)
            v_t = v_exp[start:end].unsqueeze(0).transpose(1, 2)
            o = torch.nn.functional.scaled_dot_product_attention(
                q_t, k_t, v_t, scale=self.softmax_scale, is_causal=True,
            )
            outputs.append(o.transpose(1, 2).squeeze(0))
        return torch.cat(outputs, dim=0)

    def _forward_decode(self, q, kv_cache, ctx):
        """Decode using FlashMLA (no absorption needed for V4)."""
        cache_seqlens = ctx.context_lens
        block_table = ctx.block_tables

        # V4 does NOT absorb Q into latent space (unlike V3).
        # Instead, Q is full (num_heads, head_dim=512) and we use FlashMLA
        # with the full cache directly.
        q_mla = q.unsqueeze(1)  # (N, 1, num_heads, head_dim)

        tile_sched_meta, _ = self.get_metadata(
            cache_seqlens, self.num_local_heads, num_heads_k=1,
        )

        o, _ = self.decode_op(
            q_mla,
            kv_cache.unsqueeze(-2),  # (blocks, block_size, 1, cache_dim)
            block_table,
            cache_seqlens,
            head_dim_v=_MLA_HEAD_DIM_V,
            tile_scheduler_metadata=tile_sched_meta,
            softmax_scale=self.softmax_scale,
            causal=True,
        )
        return o.reshape(-1, o.shape[-2], o.shape[-1])

    def _forward_mixed(self, q, kv_full, kv_cache, ctx):
        """Mixed prefill + decode batch."""
        np_ = ctx.num_prefill_tokens
        nd_ = ctx.num_decode_tokens

        out = q.new_empty(np_ + nd_, self.num_local_heads, self.head_dim)

        if np_ > 0:
            q_pf = q[:np_]
            kv_pf = kv_full[:np_]

            k_exp = kv_pf.unsqueeze(1).expand(-1, self.num_local_heads, -1)
            v_exp = k_exp

            cu = ctx.prefill_cu_seqlens_q
            if cu is None:
                cu = torch.tensor([0, np_], dtype=torch.int32, device=q.device)

            pf_parts = []
            for s in range(cu.shape[0] - 1):
                si, ei = cu[s].item(), cu[s + 1].item()
                q_t = q_pf[si:ei].unsqueeze(0).transpose(1, 2)
                k_t = k_exp[si:ei].unsqueeze(0).transpose(1, 2)
                v_t = v_exp[si:ei].unsqueeze(0).transpose(1, 2)
                o = torch.nn.functional.scaled_dot_product_attention(
                    q_t, k_t, v_t, scale=self.softmax_scale, is_causal=True,
                )
                pf_parts.append(o.transpose(1, 2).squeeze(0))
            o_pf = torch.cat(pf_parts, dim=0)
            out[:np_] = o_pf.view(np_, self.num_local_heads, self.head_dim)

        if nd_ > 0:
            q_dec = q[np_:]
            cache_seqlens = ctx.decode_context_lens
            block_table = ctx.decode_block_tables

            q_mla = q_dec.unsqueeze(1)
            tile_sched_meta, _ = self.get_metadata(
                cache_seqlens, self.num_local_heads, num_heads_k=1,
            )
            o, _ = self.decode_op(
                q_mla,
                kv_cache.unsqueeze(-2),
                block_table,
                cache_seqlens,
                head_dim_v=_MLA_HEAD_DIM_V,
                tile_scheduler_metadata=tile_sched_meta,
                softmax_scale=self.softmax_scale,
                causal=True,
            )
            out[np_:] = o.reshape(-1, o.shape[-2], o.shape[-1])

        return out
