"""DeepSeek Sparse Attention (DSA) Indexer for V3.2.

Lightweight FP8 MQA scorer that selects top-k KV positions per query token.
Has its own separate paged KV cache, RoPE, and Q/K projections.
Uses DeepGEMM's fp8_mqa_logits / fp8_paged_mqa_logits for scoring.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

import deep_gemm

from ....infra.context import get_context, get_attn_backend_config
from ..L1.fp8_linear import Fp8Linear
from .parallel_linear import ColumnParallelLinear

_NUM_SMS = None

_topk_ops = None


def _get_topk_ops():
    global _topk_ops
    if _topk_ops is None:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(__file__), "csrc", "top_k_per_row.cu")
        _topk_ops = load(
            name="top_k_per_row_ops",
            sources=[src],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
            verbose=False,
        )
    return _topk_ops


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


@triton.jit
def _gather_indexer_k_fp8_kernel(
    cache_ptr,              # flat [num_blocks * block_size * BYTES_PER_TOKEN] uint8
    dst_k_ptr,              # [total_tokens, head_dim] fp8 (as uint8)
    dst_scale_ptr,          # [total_tokens] float32
    token_to_seq_ptr,       # [total_tokens] int32 — maps token to sequence id
    block_table_ptr,        # [batch, max_blocks_per_seq] int32
    cu_seq_lens_ptr,        # [batch+1] int32
    block_table_stride,     # stride(0) of block_table
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BYTES_PER_TOKEN: tl.constexpr,
):
    """Gather FP8 K + scale from paged indexer cache into dense buffers."""
    tid = tl.program_id(0)

    seq_id = tl.load(token_to_seq_ptr + tid)
    seq_start = tl.load(cu_seq_lens_ptr + seq_id)

    local_pos = tid - seq_start
    page_idx = local_pos // BLOCK_SIZE
    offset_in_page = local_pos % BLOCK_SIZE

    block_id = tl.load(block_table_ptr + seq_id * block_table_stride + page_idx)
    slot = block_id * BLOCK_SIZE + offset_in_page

    src_base = slot * BYTES_PER_TOKEN
    offs = tl.arange(0, HEAD_DIM)
    fp8_bytes = tl.load(cache_ptr + src_base + offs)
    tl.store(dst_k_ptr + tid * HEAD_DIM + offs, fp8_bytes)

    scale_ptr = (cache_ptr + src_base + HEAD_DIM).to(tl.pointer_type(tl.float32))
    scale_val = tl.load(scale_ptr)
    tl.store(dst_scale_ptr + tid, scale_val)


def _build_token_to_seq(cu_seq_lens: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """Build a flat mapping from token index -> sequence index (vectorized)."""
    batch_size = cu_seq_lens.shape[0] - 1
    seq_lens = cu_seq_lens[1:] - cu_seq_lens[:-1]
    return torch.repeat_interleave(
        torch.arange(batch_size, device=cu_seq_lens.device, dtype=torch.int32),
        seq_lens.int(),
    )


def _gather_indexer_k_fp8(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    total_tokens: int,
    head_dim: int,
    block_size: int,
    dst_k: torch.Tensor | None = None,
    dst_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather FP8 K and scales from paged cache into dense contiguous buffers.

    Args:
        cache: [num_blocks, block_size, head_dim+4] uint8
        block_table: [batch, max_blocks] int32
        cu_seq_lens: [batch+1] int32 — cumulative sequence lengths
        total_tokens: total number of tokens across all sequences
    Returns:
        (k_fp8 [total_tokens, head_dim] fp8, k_scale [total_tokens] float32)
    """
    if dst_k is None:
        dst_k = torch.empty(total_tokens, head_dim, dtype=torch.uint8, device=cache.device)
    if dst_scale is None:
        dst_scale = torch.empty(total_tokens, dtype=torch.float32, device=cache.device)
    bytes_per_token = head_dim + 4

    token_to_seq = _build_token_to_seq(cu_seq_lens, total_tokens)

    _gather_indexer_k_fp8_kernel[(total_tokens,)](
        cache.view(-1),
        dst_k,
        dst_scale,
        token_to_seq,
        block_table,
        cu_seq_lens,
        block_table.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        BYTES_PER_TOKEN=bytes_per_token,
    )
    return dst_k.view(torch.float8_e4m3fn), dst_scale


def _kv_spans_from_batches(
    cu_q_list: list[int], cu_k_list: list[int], N: int, device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-query-token causal KV spans (vectorized, no Python loop).

    Following vLLM's kv_spans_from_batches pattern: uses cumsum +
    repeat_interleave to avoid per-sequence Python loops and GPU-CPU sync.
    """
    num_seqs = len(cu_q_list) - 1
    q_lens_t = torch.tensor(
        [cu_q_list[s + 1] - cu_q_list[s] for s in range(num_seqs)],
        dtype=torch.int32, device=device)
    kv_lens_t = torch.tensor(
        [cu_k_list[s + 1] - cu_k_list[s] for s in range(num_seqs)],
        dtype=torch.int32, device=device)
    cached_lens = kv_lens_t - q_lens_t

    kv_bases = torch.zeros(num_seqs, dtype=torch.int32, device=device)
    kv_bases[1:] = torch.cumsum(kv_lens_t[:-1], dim=0)

    kv_base_per_tok = torch.repeat_interleave(kv_bases, q_lens_t)
    cached_per_tok = torch.repeat_interleave(cached_lens, q_lens_t)
    kv_len_per_tok = torch.repeat_interleave(kv_lens_t, q_lens_t)

    q_starts = torch.repeat_interleave(
        torch.tensor(cu_q_list[:-1], dtype=torch.int32, device=device), q_lens_t)
    local_pos = torch.arange(N, dtype=torch.int32, device=device) - q_starts

    cu_seqlen_ks = kv_base_per_tok
    cu_seqlen_ke = (local_pos + 1 + kv_base_per_tok + cached_per_tok).clamp(
        max=(kv_base_per_tok + kv_len_per_tok))

    return cu_seqlen_ks, cu_seqlen_ke


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
        max_model_len: int = 4096,
    ):
        super().__init__()
        self.n_head = index_n_heads
        self.head_dim = index_head_dim
        self.rope_dim = qk_rope_head_dim
        self.topk_tokens = index_topk
        self.quant_block_size = 128
        self.softmax_scale = index_head_dim ** -0.5
        self.rotary_emb = rotary_emb
        self.max_model_len = max_model_len

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

        self._k_fp8_buf = None
        self._k_scale_buf = None
        self._k_buf_size = 0

    def _get_gather_bufs(self, size: int, device):
        """Return reusable (k_fp8, k_scale) buffers of at least [size, head_dim]."""
        if self._k_fp8_buf is None or self._k_buf_size < size:
            self._k_buf_size = max(size, self._k_buf_size * 2, 256)
            self._k_fp8_buf = torch.empty(
                self._k_buf_size, self.head_dim,
                dtype=torch.uint8, device=device,
            )
            self._k_scale_buf = torch.empty(
                self._k_buf_size, dtype=torch.float32, device=device,
            )
        return self._k_fp8_buf[:size], self._k_scale_buf[:size]

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
            self._prefill_indexing(q_fp8[:np_], q_scale[:np_], weights[:np_],
                                   topk_indices_buffer, ctx)

        if not ctx.is_prefill or (ctx.is_mixed and ctx.num_decode_tokens > 0):
            nd_start = ctx.num_prefill_tokens if ctx.is_mixed else 0
            nd_end = N
            if nd_end > nd_start:
                self._decode_indexing(
                    q_fp8[nd_start:nd_end], q_scale[nd_start:nd_end],
                    weights[nd_start:nd_end],
                    topk_indices_buffer, nd_start, ctx,
                )

        return topk_indices_buffer

    def _prefill_indexing(self, q_fp8, q_scale, weights, topk_indices_buffer, ctx):
        """Prefill: fused FP8 MQA logits via DeepGEMM with causal masking."""
        if not self.k_cache.numel():
            return

        if ctx.is_mixed:
            cu_q = ctx.prefill_cu_seqlens_q
            cu_k = ctx.prefill_cu_seqlens_k
            block_tables = ctx.prefill_block_tables
        else:
            cu_q = ctx.cu_seqlens_q
            cu_k = ctx.cu_seqlens_k
            block_tables = ctx.block_tables

        cu_q_list = (ctx.prefill_cu_seqlens_q_cpu if ctx.is_mixed and ctx.prefill_cu_seqlens_q_cpu is not None
                     else ctx.cu_seqlens_q_cpu if not ctx.is_mixed and ctx.cu_seqlens_q_cpu is not None
                     else cu_q.cpu().tolist())
        cu_k_list = (ctx.prefill_cu_seqlens_k_cpu if ctx.is_mixed and ctx.prefill_cu_seqlens_k_cpu is not None
                     else ctx.cu_seqlens_k_cpu if not ctx.is_mixed and ctx.cu_seqlens_k_cpu is not None
                     else cu_k.cpu().tolist())
        num_seqs = len(cu_q_list) - 1
        N = q_fp8.shape[0]
        total_kv = cu_k_list[-1] - cu_k_list[0]
        device = q_fp8.device

        if total_kv == 0:
            return

        # Gather all K from paged cache into dense FP8 + scale buffers
        if block_tables is not None and block_tables.numel() > 0 and self.k_cache.numel():
            dst_k, dst_scale = self._get_gather_bufs(total_kv, device)
            cu_k_gather = torch.tensor(cu_k_list, dtype=torch.int32, device=device)
            k_fp8_dense, k_scale_dense = _gather_indexer_k_fp8(
                self.k_cache,
                block_tables[:num_seqs],
                cu_k_gather,
                total_kv,
                self.head_dim,
                self._block_size,
                dst_k=dst_k.view(torch.uint8),
                dst_scale=dst_scale,
            )
        else:
            k_fp8_dense = torch.empty(0, self.head_dim, dtype=torch.float8_e4m3fn, device=device)
            k_scale_dense = torch.empty(0, dtype=torch.float32, device=device)

        cu_seqlen_ks, cu_seqlen_ke = _kv_spans_from_batches(
            cu_q_list, cu_k_list, N, device)

        w = weights.to(torch.float32)

        max_prefill_buf = int(os.environ.get(
            "KB_NANO_INDEXER_MAX_PREFILL_TOKENS", "8192"))
        if N <= max_prefill_buf or total_kv <= max_prefill_buf:
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8,
                (k_fp8_dense, k_scale_dense),
                w,
                cu_seqlen_ks,
                cu_seqlen_ke,
                clean_logits=False,
            )
            ops = _get_topk_ops()
            ops.top_k_per_row_prefill(
                logits,
                cu_seqlen_ks,
                cu_seqlen_ke,
                topk_indices_buffer[:N],
                N,
                logits.stride(0),
                logits.stride(1),
                self.topk_tokens,
            )
        else:
            ops = _get_topk_ops()
            chunk_start = 0
            while chunk_start < N:
                chunk_end = min(chunk_start + max_prefill_buf, N)
                chunk_n = chunk_end - chunk_start

                chunk_logits = deep_gemm.fp8_mqa_logits(
                    q_fp8[chunk_start:chunk_end],
                    (k_fp8_dense, k_scale_dense),
                    w[chunk_start:chunk_end],
                    cu_seqlen_ks[chunk_start:chunk_end],
                    cu_seqlen_ke[chunk_start:chunk_end],
                    clean_logits=False,
                )
                ops.top_k_per_row_prefill(
                    chunk_logits,
                    cu_seqlen_ks[chunk_start:chunk_end],
                    cu_seqlen_ke[chunk_start:chunk_end],
                    topk_indices_buffer[chunk_start:chunk_end],
                    chunk_n,
                    chunk_logits.stride(0),
                    chunk_logits.stride(1),
                    self.topk_tokens,
                )
                del chunk_logits
                chunk_start = chunk_end

    def _decode_indexing(self, q_fp8, q_scale, weights, topk_indices_buffer,
                         offset, ctx):
        """Decode: fused FP8 paged MQA logits via DeepGEMM."""
        if not self.k_cache.numel():
            return

        if ctx.is_mixed:
            context_lens = ctx.decode_context_lens
            block_tables = ctx.decode_block_tables
        else:
            context_lens = ctx.context_lens
            block_tables = ctx.block_tables

        batch = q_fp8.shape[0]
        if batch == 0:
            return

        # q_fp8: [B, n_head, head_dim] -> [B, 1, n_head, head_dim] (next_n=1)
        q_4d = q_fp8.unsqueeze(1)

        # kv_cache: [num_blocks, block_size, D+4] -> [num_blocks, block_size, 1, D+4]
        kv_cache_4d = self.k_cache.unsqueeze(2)

        # weights already has q_scale * softmax_scale * n_head^-0.5 applied
        w = weights.to(torch.float32)

        ctx_lens = context_lens[:batch].to(torch.int32)

        if ctx.decode_paged_mqa_sched_meta is not None:
            sched_meta = ctx.decode_paged_mqa_sched_meta
        else:
            global _NUM_SMS
            if _NUM_SMS is None:
                _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
            sched_meta = deep_gemm.get_paged_mqa_logits_metadata(
                ctx_lens, self._block_size, _NUM_SMS,
            )
            ctx.decode_paged_mqa_sched_meta = sched_meta

        logits = deep_gemm.fp8_paged_mqa_logits(
            q_4d,
            kv_cache_4d,
            w,
            ctx_lens,
            block_tables[:batch].to(torch.int32),
            sched_meta,
            self.max_model_len,
            clean_logits=False,
        )

        ops = _get_topk_ops()
        ops.top_k_per_row_decode(
            logits,
            1,
            ctx_lens,
            topk_indices_buffer[offset:offset + batch],
            batch,
            logits.stride(0),
            logits.stride(1),
            self.topk_tokens,
        )

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
