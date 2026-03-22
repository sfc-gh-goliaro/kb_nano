"""DeepSeek Sparse Attention (DSA) Indexer for V3.2.

Lightweight FP8 MQA scorer that selects top-k KV positions per query token.
Has its own separate paged KV cache, RoPE, and Q/K projections.
Uses DeepGEMM's fp8_mqa_logits / fp8_paged_mqa_logits for scoring.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.context import get_context, get_attn_backend_config
from ..L1.fp8_linear import Fp8Linear
from .parallel_linear import ColumnParallelLinear


class _IndexerLinear(nn.Module):
    """FP8 replicated linear for indexer projections."""

    def __init__(self, in_features: int, out_features: int, quant_config: dict | None = None):
        super().__init__()
        _FP8_BLOCK = 128
        self.use_fp8 = quant_config is not None
        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(
                    math.ceil(out_features / _FP8_BLOCK),
                    math.ceil(in_features / _FP8_BLOCK),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)
            self.weight_scale_inv.weight_loader = lambda p, w: p.data.copy_(w)
            self.linear_op = Fp8Linear()
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)
            from ..L1.linear import Linear
            self.linear_op = Linear()

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, None)
        return self.linear_op(x, self.weight, None)


@triton.jit
def _per_token_group_quant_fp8_kernel(
    x_ptr, x_stride,
    out_ptr, out_stride,
    scale_ptr, scale_stride,
    GROUP_SIZE: tl.constexpr,
    N_GROUPS: tl.constexpr,
):
    """Quantize input to FP8 per-token-group with UE8M0 scales."""
    row = tl.program_id(0)
    group = tl.program_id(1)
    offs = tl.arange(0, GROUP_SIZE)
    src = tl.load(x_ptr + row * x_stride + group * GROUP_SIZE + offs).to(tl.float32)
    amax = tl.max(tl.abs(src))
    scale_inv = amax / 448.0
    scale_inv = tl.where(scale_inv < 1e-12, 1e-12, scale_inv)
    log2_s = tl.math.log2(scale_inv)
    log2_s_ceil = tl.math.ceil(log2_s)
    scale_inv_ue8m0 = tl.math.exp2(log2_s_ceil)
    fp8_vals = (src / scale_inv_ue8m0).to(tl.float8e4nv)
    tl.store(out_ptr + row * out_stride + group * GROUP_SIZE + offs, fp8_vals)
    tl.store(scale_ptr + row * scale_stride + group, scale_inv_ue8m0)


def _per_token_group_quant_fp8(x: torch.Tensor, group_size: int = 128):
    """Quantize to FP8 with per-token-group UE8M0 scales.

    Args:
        x: [M, D] bfloat16
    Returns:
        (x_fp8 [M, D] float8_e4m3fn, scales [M, D//group_size] float32)
    """
    M, D = x.shape
    n_groups = D // group_size
    x_fp8 = torch.empty(M, D, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty(M, n_groups, dtype=torch.float32, device=x.device)
    _per_token_group_quant_fp8_kernel[(M, n_groups)](
        x, x.stride(0),
        x_fp8, x_fp8.stride(0),
        scales, scales.stride(0),
        GROUP_SIZE=group_size,
        N_GROUPS=n_groups,
    )
    return x_fp8, scales


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr, k_stride,           # [num_tokens, head_dim] bf16
    cache_ptr,                 # flat [num_blocks * block_size * (head_dim + 4)] uint8
    slot_mapping_ptr,          # [num_tokens] int64
    HEAD_DIM: tl.constexpr,    # 128
    QUANT_BLOCK: tl.constexpr, # 128
    BYTES_PER_TOKEN: tl.constexpr,  # head_dim + 4 (scale as float32)
):
    """Quantize K to FP8 and store into indexer paged cache."""
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offs = tl.arange(0, HEAD_DIM)
    vals = tl.load(k_ptr + idx * k_stride + offs).to(tl.float32)
    amax = tl.max(tl.abs(vals))
    scale_inv = amax / 448.0
    scale_inv = tl.where(scale_inv < 1e-12, 1e-12, scale_inv)
    log2_s = tl.math.log2(scale_inv)
    scale_inv_ue8m0 = tl.math.exp2(tl.math.ceil(log2_s))
    fp8_vals = (vals / scale_inv_ue8m0).to(tl.float8e4nv)
    fp8_u8 = fp8_vals.to(tl.uint8, bitcast=True)
    base = slot * BYTES_PER_TOKEN
    tl.store(cache_ptr + base + offs, fp8_u8)
    scale_byte_ptr = (cache_ptr + base + HEAD_DIM).to(tl.pointer_type(tl.float32))
    tl.store(scale_byte_ptr, scale_inv_ue8m0)


class DeepSeekIndexer(nn.Module):
    """DSA Indexer: computes top-k KV indices per query token per layer."""

    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        index_n_heads: int,
        index_head_dim: int,
        qk_rope_head_dim: int,
        index_topk: int,
        rms_norm_eps: float,
        rotary_emb: nn.Module,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.n_head = index_n_heads
        self.head_dim = index_head_dim
        self.rope_dim = qk_rope_head_dim
        self.topk_tokens = index_topk
        self.quant_block_size = 128
        self.softmax_scale = index_head_dim ** -0.5
        self.rotary_emb = rotary_emb

        self.wq_b = _IndexerLinear(q_lora_rank, index_n_heads * index_head_dim,
                                   quant_config=quant_config)
        self.wk = _IndexerLinear(hidden_size, index_head_dim,
                                 quant_config=quant_config)
        self.k_norm = nn.LayerNorm(index_head_dim, eps=1e-6)
        self.weights_proj = _IndexerLinear(hidden_size, index_n_heads,
                                          quant_config=None)

        self.k_cache = torch.tensor([])
        self._block_size = get_attn_backend_config().block_size

        self._bytes_per_token = index_head_dim + 4

    def forward(self, hidden_states, q_c, positions, topk_indices_buffer):
        """
        Args:
            hidden_states: [N, hidden_size]
            q_c: [N, q_lora_rank] (compressed Q from q_a_proj + layernorm)
            positions: [N]
            topk_indices_buffer: [max_tokens, topk_tokens] int32 (shared, written in-place)
        Returns:
            topk_indices_buffer (modified in-place)
        """
        ctx = get_context()
        N = hidden_states.shape[0]

        q = self.wq_b(q_c)
        q = q.view(N, self.n_head, self.head_dim)
        q_pe = q[..., :self.rope_dim]
        q_nope = q[..., self.rope_dim:]

        k = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe = k[..., :self.rope_dim]
        k_nope = k[..., self.rope_dim:]

        q_pe_flat = q_pe.reshape(N, self.n_head * self.rope_dim)
        k_pe_flat = k_pe
        q_pe_flat, k_pe_flat = self.rotary_emb(positions, q_pe_flat, k_pe_flat)
        q_pe = q_pe_flat.view(N, self.n_head, self.rope_dim)
        k_pe = k_pe_flat.view(N, self.rope_dim)

        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)

        q_flat = q.reshape(N * self.n_head, self.head_dim)
        q_fp8, q_scale = _per_token_group_quant_fp8(q_flat, self.quant_block_size)
        q_fp8 = q_fp8.view(N, self.n_head, self.head_dim)
        q_scale = q_scale.view(N, self.n_head, -1)

        weights = self.weights_proj(hidden_states)
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head ** -0.5
        weights = weights.squeeze(-1)

        if self.k_cache.numel():
            _indexer_k_quant_and_cache_kernel[(N,)](
                k, k.stride(0),
                self.k_cache.view(-1),
                ctx.slot_mapping,
                HEAD_DIM=self.head_dim,
                QUANT_BLOCK=self.quant_block_size,
                BYTES_PER_TOKEN=self._bytes_per_token,
            )

        topk_indices_buffer[:N] = -1

        if ctx.is_prefill or (ctx.is_mixed and ctx.num_prefill_tokens > 0):
            np_ = ctx.num_prefill_tokens if ctx.is_mixed else N
            self._prefill_indexing(q_fp8[:np_], weights[:np_], topk_indices_buffer, ctx)

        if not ctx.is_prefill or (ctx.is_mixed and ctx.num_decode_tokens > 0):
            nd_start = ctx.num_prefill_tokens if ctx.is_mixed else 0
            nd_end = N
            if nd_end > nd_start:
                self._decode_indexing(
                    q_fp8[nd_start:nd_end], weights[nd_start:nd_end],
                    topk_indices_buffer, nd_start, ctx,
                )

        return topk_indices_buffer

    def _prefill_indexing(self, q_fp8, weights, topk_indices_buffer, ctx):
        """Prefill: gather all K from cache, compute logits, top-k."""
        if ctx.is_mixed:
            cu_q = ctx.prefill_cu_seqlens_q
            cu_k = ctx.prefill_cu_seqlens_k
        else:
            cu_q = ctx.cu_seqlens_q
            cu_k = ctx.cu_seqlens_k

        cu_q_list = cu_q.cpu().tolist()
        cu_k_list = cu_k.cpu().tolist()
        num_seqs = len(cu_q_list) - 1
        N = q_fp8.shape[0]

        for s in range(num_seqs):
            q_start, q_end = cu_q_list[s], cu_q_list[s + 1]
            kv_len = cu_k_list[s + 1] - cu_k_list[s]
            if kv_len == 0:
                continue

            q_s = q_fp8[q_start:q_end]
            q_flat = q_s.reshape(-1, self.head_dim).to(torch.bfloat16)
            w_s = weights[q_start:q_end]

            k_latent = self._gather_k_bf16(ctx, s, kv_len)
            k_latent = k_latent.reshape(kv_len, self.head_dim)
            logits = torch.matmul(q_flat, k_latent.t())
            logits = logits.view(q_end - q_start, self.n_head, kv_len)
            logits = logits + w_s.unsqueeze(-1)

            logits_combined = logits.sum(dim=1)

            q_len = q_end - q_start
            cached_len = kv_len - q_len

            causal_mask = torch.arange(kv_len, device=logits_combined.device).unsqueeze(0)
            valid_lens = torch.arange(1, q_len + 1, device=logits_combined.device) + cached_len
            valid_lens = valid_lens.clamp(max=kv_len)
            mask = causal_mask >= valid_lens.unsqueeze(1)
            logits_combined = logits_combined.masked_fill(mask, float('-inf'))

            k = min(self.topk_tokens, kv_len)
            if k > 0:
                _, indices = torch.topk(logits_combined, k, dim=1)
                topk_indices_buffer[q_start:q_end, :k] = indices.to(torch.int32)

    def _decode_indexing(self, q_fp8, weights, topk_indices_buffer, offset, ctx):
        """Decode: paged MQA logits via simple matmul fallback."""
        if ctx.is_mixed:
            context_lens = ctx.decode_context_lens
            block_tables = ctx.decode_block_tables
        else:
            context_lens = ctx.context_lens
            block_tables = ctx.block_tables

        batch = q_fp8.shape[0]

        for b in range(batch):
            seq_len = int(context_lens[b].item())
            if seq_len == 0:
                continue

            q_b = q_fp8[b].reshape(self.n_head, self.head_dim).to(torch.bfloat16)
            w_b = weights[b]

            num_pages = (seq_len + self._block_size - 1) // self._block_size
            bt = block_tables[b, :num_pages]
            k_gathered = self._gather_k_bf16_paged(bt, seq_len)
            k_gathered = k_gathered.reshape(seq_len, self.head_dim)
            logits = torch.matmul(q_b, k_gathered.t())
            logits = logits + w_b.unsqueeze(-1)
            logits_combined = logits.sum(dim=0)

            k = min(self.topk_tokens, seq_len)
            _, indices = torch.topk(logits_combined[:seq_len], k)
            topk_indices_buffer[offset + b, :k] = indices.to(torch.int32)

    def _gather_k_bf16(self, ctx, seq_idx, kv_len):
        """Gather K from indexer cache pages, dequantize to BF16."""
        if ctx.is_mixed:
            bt = ctx.prefill_block_tables
        elif hasattr(ctx, 'block_tables') and ctx.block_tables is not None:
            bt = ctx.block_tables
        else:
            return torch.zeros(kv_len, self.head_dim, dtype=torch.bfloat16,
                               device=self.k_cache.device)

        if bt is None or bt.numel() == 0:
            return torch.zeros(kv_len, self.head_dim, dtype=torch.bfloat16,
                               device=self.k_cache.device)

        num_pages = (kv_len + self._block_size - 1) // self._block_size
        page_ids = bt[seq_idx, :num_pages]
        return self._gather_k_bf16_paged(page_ids, kv_len)

    def _gather_k_bf16_paged(self, page_ids, total_tokens):
        """Gather and dequantize K from indexer cache pages."""
        cache = self.k_cache
        hd = self.head_dim
        bs = self._block_size

        if total_tokens == 0:
            return torch.zeros(0, hd, dtype=torch.bfloat16, device=cache.device)

        num_pages = page_ids.shape[0]
        raw = cache[page_ids.long()]  # (num_pages, block_size, bytes_per_token)

        flat = raw.reshape(num_pages * bs, -1)[:total_tokens]
        fp8_vals = flat[:, :hd].contiguous().view(torch.float8_e4m3fn).to(torch.float32)
        scales = flat[:, hd:hd + 4].contiguous().view(torch.float32).view(total_tokens)
        return (fp8_vals * scales.unsqueeze(-1)).to(torch.bfloat16)
