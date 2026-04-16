"""vLLM-aligned Attention layer with paged KV cache.

Mirrors vLLM's ``Attention`` class (from
``vllm/model_executor/layers/attention/attention.py``):

    forward(query, key, value) -> torch.Tensor

Inputs and outputs are **flat** ``[N, num_heads * head_dim]`` tensors.
KV cache metadata is obtained from the global ``Context`` (via
``get_context()``), matching vLLM's ``get_forward_context()`` pattern.

Backend selection (flash_attn vs TRTLLM-gen) is handled at init time
via ``AttnBackendConfig``.  The engine discovers this module for KV cache
assignment through duck-typing (``hasattr(module, "k_cache")``).

TODO(tech-debt): CUDA graph capture is incompatible with chunked local
attention because the metadata remapping (cu_seqlens, block_tables) varies
per batch.  vLLM disables CUDA graphs when chunked local attention is
active.  If/when we add CUDA graph support, we need to handle this case.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....infra.context import get_context, get_attn_backend_config
from ..L1.store_kvcache import StoreKVCache, StoreKVCacheHND


def _chunked_prefill_remap(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor | None,
    attention_chunk_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor | None]:
    """Remap prefill metadata into chunked local-attention virtual batches.

    Follows vLLM's ``make_local_attention_virtual_batches`` algorithm: each
    original sequence is split into ``attention_chunk_size``-wide chunks that
    the kernel sees as independent sequences.

    Returns (cu_seqlens_q', cu_seqlens_k', max_seqlen_q', max_seqlen_k',
             block_tables').
    """
    device = cu_seqlens_q.device
    cu_q_np = cu_seqlens_q.cpu().numpy()
    cu_k_np = cu_seqlens_k.cpu().numpy()

    q_seqlens = cu_q_np[1:] - cu_q_np[:-1]
    k_seqlens = cu_k_np[1:] - cu_k_np[:-1]
    batch_size = len(q_seqlens)

    q_tokens_in_first_block = np.minimum(
        attention_chunk_size - ((k_seqlens - q_seqlens) % attention_chunk_size),
        q_seqlens,
    ).astype(np.int32)
    tokens_in_last_block = (
        attention_chunk_size + (k_seqlens % -attention_chunk_size)
    ).astype(np.int32)

    local_blocks = (
        1 + np.ceil(
            np.maximum(q_seqlens - q_tokens_in_first_block, 0) / attention_chunk_size
        ).astype(np.int32)
    )

    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = int(cu_num_blocks[-1])

    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1

    seqlens_q_local = np.repeat(
        q_seqlens - q_tokens_in_first_block, local_blocks,
    ).astype(np.int32)
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attention_chunk_size * (arange - 1),
        attention_chunk_size,
    )[arange > 0]

    cu_seqlens_q_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_q_local, out=cu_seqlens_q_local[1:])
    cu_seqlens_q_local[0] = 0

    seqlens_k_local = np.full(virtual_batches, attention_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block

    cu_seqlens_k_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_k_local, out=cu_seqlens_k_local[1:])
    cu_seqlens_k_local[0] = 0

    max_seqlen_q = int(seqlens_q_local.max()) if virtual_batches > 0 else 0
    max_seqlen_k = int(seqlens_k_local.max()) if virtual_batches > 0 else 0

    cu_q_out = torch.from_numpy(cu_seqlens_q_local).to(device=device)
    cu_k_out = torch.from_numpy(cu_seqlens_k_local).to(device=device)

    block_tables_out = None
    if block_tables is not None and block_size > 0:
        assert attention_chunk_size % block_size == 0
        pages_per_chunk = attention_chunk_size // block_size

        k_seqstarts_absolute = np.repeat(k_seqlens, local_blocks) - (
            rarange * attention_chunk_size
            + np.repeat(tokens_in_last_block, local_blocks)
        )
        block_starts = k_seqstarts_absolute // block_size

        block_indices = (
            block_starts[:, None]
            + np.arange(pages_per_chunk, dtype=np.int32)
        )
        block_indices = block_indices.reshape(-1).clip(
            max=block_tables.shape[1] - 1,
        )
        batch_indices = np.repeat(
            np.arange(batch_size, dtype=np.int32),
            local_blocks * pages_per_chunk,
        )

        bi_torch = torch.from_numpy(batch_indices)
        bk_torch = torch.from_numpy(block_indices)
        block_tables_out = block_tables[bi_torch, bk_torch].view(
            virtual_batches, -1,
        )

    return cu_q_out, cu_k_out, max_seqlen_q, max_seqlen_k, block_tables_out


def _chunked_decode_remap(
    cache_seqlens: torch.Tensor,
    block_tables: torch.Tensor | None,
    attention_chunk_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Remap decode metadata so the kernel only attends within the last chunk.

    Returns (cache_seqlens', block_tables', max_context_len').
    """
    local_seqlens = torch.clamp(cache_seqlens, max=attention_chunk_size)
    max_context_len = int(local_seqlens.max().item()) if local_seqlens.numel() > 0 else 0

    if block_tables is not None and block_size > 0:
        assert attention_chunk_size % block_size == 0
        pages_per_chunk = attention_chunk_size // block_size
        chunk_start_page = (cache_seqlens - local_seqlens) // block_size
        offsets = torch.arange(pages_per_chunk, device=block_tables.device)
        page_indices = chunk_start_page.unsqueeze(1) + offsets
        page_indices = page_indices.clamp(max=block_tables.shape[1] - 1)
        block_tables = torch.gather(block_tables, 1, page_indices)

    return local_seqlens, block_tables, max_context_len


class Attention(nn.Module):

    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int | None = None,
                 sliding_window: int | None = None,
                 sinks: torch.nn.Parameter | None = None,
                 attention_chunk_size: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window
        self.sinks = sinks
        self.attention_chunk_size = attention_chunk_size

        # TODO(tech-debt): For chunked local attention layers the KV cache
        # could be limited to ``attention_chunk_size`` tokens per layer instead
        # of ``max_seq_len``, following vLLM's ``ChunkedLocalAttentionSpec``.
        # This is not needed for correctness but would reduce memory usage.
        self.k_cache = self.v_cache = torch.tensor([])

        attn_cfg = get_attn_backend_config()
        self._use_trtllm = attn_cfg.use_trtllm
        self._use_sdpa = False
        self._block_size = attn_cfg.block_size

        # FA3 native sinks: pass sinks as s_aux, sliding window as window_size
        self._fa3_sinks = sinks
        self._fa3_window_size = (
            (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
        )

        self._use_custom_op = False
        self._layer_name = ""

        if self._use_trtllm:
            self.store_kvcache = StoreKVCacheHND(page_size=attn_cfg.block_size)
            from ..L1.flashinfer_prefill import TRTLLMPrefill
            from ..L1.flashinfer_decode import TRTLLMDecode
            self.prefill_op = TRTLLMPrefill(
                self.num_heads, self.num_kv_heads, head_size,
            )
            self.decode_op = TRTLLMDecode(
                self.num_heads, self.num_kv_heads, head_size,
            )
        else:
            self.store_kvcache = StoreKVCache()
            from ..L1.flash_attn_prefill import FlashAttnPrefill
            from ..L1.flash_attn_decode import FlashAttnDecode
            self.prefill_op = FlashAttnPrefill(
                self.num_heads, self.num_kv_heads, head_size,
            )
            self.decode_op = FlashAttnDecode(
                self.num_heads, self.num_kv_heads, head_size,
            )

        # FlashAttn path now handles sinks via s_aux, no SDPA fallback needed

    def set_trtllm_workspace(self, workspace: torch.Tensor):
        if self._use_trtllm:
            self.decode_op._workspace = workspace
            self.prefill_op._workspace = workspace

    def forward_impl(self, query: torch.Tensor, key: torch.Tensor,
                     value: torch.Tensor) -> torch.Tensor:
        """Core attention logic, callable from both eager and custom-op paths."""
        ctx = get_context()
        N = query.shape[0]

        q = query.view(N, self.num_heads, self.head_size)
        k = key.view(N, self.num_kv_heads, self.head_size)
        v = value.view(N, self.num_kv_heads, self.head_size)

        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.store_kvcache(k, v, k_cache, v_cache, ctx.slot_mapping)

        if ctx.is_mixed:
            o = self._forward_mixed(q, k_cache, v_cache, ctx)
        else:
            o = self._forward_pure(q, k, v, k_cache, v_cache, ctx)

        return o.reshape(N, self.num_heads * self.head_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            return torch.ops.kb_nano.unified_attention(
                query, key, value, self._layer_name,
            )
        return self.forward_impl(query, key, value)

    # ---- SDPA path (for GPT-OSS attention sinks) ----

    def _forward_sdpa(self, q, k, v, k_cache, v_cache, ctx):
        """SDPA-based forward path for attention with sinks and/or sliding window."""
        if ctx.is_mixed:
            return self._forward_sdpa_mixed(q, k_cache, v_cache, ctx)
        if ctx.is_prefill:
            return self._forward_sdpa_prefill(q, k, v, ctx)
        return self._forward_sdpa_decode(q, k_cache, v_cache, ctx)

    def _forward_sdpa_mixed(self, q, k_cache, v_cache, ctx):
        """Handle mixed prefill+decode batch for SDPA path."""
        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)

        if np_ > 0:
            cu_q = ctx.prefill_cu_seqlens_q
            B_pf = cu_q.shape[0] - 1
            outputs = []
            for b in range(B_pf):
                start = int(cu_q[b].item())
                end = int(cu_q[b + 1].item())
                T = end - start
                q_b = q[start:end]

                seq_len = int(ctx.prefill_cu_seqlens_k[b + 1].item() - ctx.prefill_cu_seqlens_k[b].item())
                k_b, v_b = self._gather_kv_from_cache(
                    k_cache, v_cache, ctx.prefill_block_tables[b], seq_len,
                )

                if self.num_kv_heads != self.num_heads:
                    rep = self.num_heads // self.num_kv_heads
                    k_b = k_b.repeat_interleave(rep, dim=1)
                    v_b = v_b.repeat_interleave(rep, dim=1)

                if self.sinks is not None:
                    zero_kv = torch.zeros(1, self.num_heads, self.head_size,
                                          device=q.device, dtype=q.dtype)
                    k_ext = torch.cat([k_b, zero_kv], dim=0)
                    v_ext = torch.cat([v_b, zero_kv], dim=0)
                    q_4d = q_b.transpose(0, 1).unsqueeze(0)
                    k_4d = k_ext.transpose(0, 1).unsqueeze(0)
                    v_4d = v_ext.transpose(0, 1).unsqueeze(0)
                    attn_bias = self._make_prefill_bias(T, seq_len, q.device, q.dtype)
                    o_b = F.scaled_dot_product_attention(
                        q_4d, k_4d, v_4d, attn_mask=attn_bias.unsqueeze(0),
                        scale=self.scale,
                    )
                    outputs.append(o_b.squeeze(0).transpose(0, 1))
                else:
                    q_4d = q_b.transpose(0, 1).unsqueeze(0)
                    k_4d = k_b.transpose(0, 1).unsqueeze(0)
                    v_4d = v_b.transpose(0, 1).unsqueeze(0)
                    o_b = F.scaled_dot_product_attention(
                        q_4d, k_4d, v_4d, is_causal=(T == seq_len), scale=self.scale,
                    )
                    outputs.append(o_b.squeeze(0).transpose(0, 1))
            out[:np_] = torch.cat(outputs, dim=0)

        if nd > 0:
            decode_ctx_wrapper = type(ctx)(
                block_tables=ctx.decode_block_tables,
                context_lens=ctx.decode_context_lens,
                max_context_len=ctx.decode_max_context_len,
            )
            out[np_:] = self._forward_sdpa_decode(q[np_:], k_cache, v_cache, decode_ctx_wrapper)

        return out

    def _gather_kv_from_cache(self, k_cache, v_cache, block_table, seq_len):
        """Gather K/V from paged cache for a single sequence."""
        block_size = k_cache.shape[1]
        num_blocks = (seq_len + block_size - 1) // block_size
        k_out = torch.zeros(seq_len, self.num_kv_heads, self.head_size,
                            device=k_cache.device, dtype=k_cache.dtype)
        v_out = torch.zeros_like(k_out)
        for blk_idx in range(num_blocks):
            block_id = block_table[blk_idx]
            start = blk_idx * block_size
            end = min(start + block_size, seq_len)
            length = end - start
            k_out[start:end] = k_cache[block_id, :length]
            v_out[start:end] = v_cache[block_id, :length]
        return k_out, v_out

    def _forward_sdpa_prefill(self, q, k, v, ctx):
        cu_q = ctx.cu_seqlens_q
        B = cu_q.shape[0] - 1
        outputs = []
        for b in range(B):
            start = int(cu_q[b].item())
            end = int(cu_q[b + 1].item())
            T = end - start
            q_b = q[start:end]
            k_b = k[start:end]
            v_b = v[start:end]

            if self.num_kv_heads != self.num_heads:
                rep = self.num_heads // self.num_kv_heads
                k_b = k_b.repeat_interleave(rep, dim=1)
                v_b = v_b.repeat_interleave(rep, dim=1)

            if self.sinks is not None:
                zero_kv = torch.zeros(1, self.num_heads, self.head_size,
                                      device=q.device, dtype=q.dtype)
                k_ext = torch.cat([k_b, zero_kv], dim=0)
                v_ext = torch.cat([v_b, zero_kv], dim=0)
                q_4d = q_b.transpose(0, 1).unsqueeze(0)
                k_4d = k_ext.transpose(0, 1).unsqueeze(0)
                v_4d = v_ext.transpose(0, 1).unsqueeze(0)
                attn_bias = self._make_prefill_bias(T, T, q.device, q.dtype)
                o_b = F.scaled_dot_product_attention(
                    q_4d, k_4d, v_4d, attn_mask=attn_bias.unsqueeze(0),
                    scale=self.scale,
                )
                outputs.append(o_b.squeeze(0).transpose(0, 1))
            else:
                q_4d = q_b.transpose(0, 1).unsqueeze(0)
                k_4d = k_b.transpose(0, 1).unsqueeze(0)
                v_4d = v_b.transpose(0, 1).unsqueeze(0)
                o_b = F.scaled_dot_product_attention(
                    q_4d, k_4d, v_4d, is_causal=True, scale=self.scale,
                )
                outputs.append(o_b.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)

    def _forward_sdpa_decode(self, q, k_cache, v_cache, ctx):
        B = ctx.block_tables.shape[0]
        max_seq = int(ctx.context_lens.max().item())
        block_size = k_cache.shape[1]

        k_gathered = torch.zeros(B, max_seq, self.num_kv_heads, self.head_size,
                                 device=k_cache.device, dtype=k_cache.dtype)
        v_gathered = torch.zeros_like(k_gathered)
        for b in range(B):
            seq_len = int(ctx.context_lens[b].item())
            num_blocks = (seq_len + block_size - 1) // block_size
            for blk_idx in range(num_blocks):
                block_id = ctx.block_tables[b, blk_idx]
                start = blk_idx * block_size
                end = min(start + block_size, seq_len)
                length = end - start
                k_gathered[b, start:end] = k_cache[block_id, :length]
                v_gathered[b, start:end] = v_cache[block_id, :length]

        q_4d = q.view(B, 1, self.num_heads, self.head_size).transpose(1, 2)
        k_4d = k_gathered.transpose(1, 2)
        v_4d = v_gathered.transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k_4d = k_4d.repeat_interleave(rep, dim=1)
            v_4d = v_4d.repeat_interleave(rep, dim=1)

        if self.sinks is not None:
            zero_kv = torch.zeros(B, self.num_heads, 1, self.head_size,
                                  device=q.device, dtype=q.dtype)
            k_4d = torch.cat([k_4d, zero_kv], dim=2)
            v_4d = torch.cat([v_4d, zero_kv], dim=2)
            attn_bias = torch.zeros(self.num_heads, 1, max_seq + 1,
                                    device=q.device, dtype=q.dtype)
            attn_bias[:, :, -1] = self.sinks.unsqueeze(1)
            if self.sliding_window is not None:
                for b_idx in range(B):
                    seq_len = int(ctx.context_lens[b_idx].item())
                    cutoff = seq_len - self.sliding_window
                    if cutoff > 0:
                        attn_bias[:, :, :cutoff] = float("-inf")
            for b_idx in range(B):
                seq_len = int(ctx.context_lens[b_idx].item())
                if seq_len < max_seq:
                    attn_bias[:, :, seq_len:max_seq] = float("-inf")
            attn_bias = attn_bias.unsqueeze(0).expand(B, -1, -1, -1)
            o = F.scaled_dot_product_attention(
                q_4d, k_4d, v_4d, attn_mask=attn_bias, scale=self.scale,
            )
        else:
            o = F.scaled_dot_product_attention(
                q_4d, k_4d, v_4d, is_causal=False, scale=self.scale,
            )
        return o.transpose(1, 2).reshape(B, self.num_heads, self.head_size)

    def _make_prefill_bias(self, T: int, S: int, device, dtype):
        """Build attention bias: causal + sliding window + virtual sink column."""
        bias = torch.zeros(T, S, device=device, dtype=dtype)
        if T > 1:
            causal = torch.triu(
                torch.full((T, S), float("-inf"), device=device, dtype=dtype),
                diagonal=S - T + 1,
            )
            bias = bias + causal
        if self.sliding_window is not None and S > self.sliding_window:
            sw_mask = torch.tril(
                torch.full((T, S), float("-inf"), device=device, dtype=dtype),
                diagonal=S - T - self.sliding_window,
            )
            bias = bias + sw_mask
        bias = bias.unsqueeze(0).expand(self.num_heads, -1, -1).contiguous()
        if self.sinks is not None:
            sink_col = self.sinks.view(self.num_heads, 1, 1).expand(-1, T, 1)
            bias = torch.cat([bias, sink_col], dim=-1)
        return bias

    # ---- FlashAttn / TRTLLM paths ----

    def _forward_pure(self, q, k, v, k_cache, v_cache, ctx):
        fa_extra = {}
        if self._fa3_sinks is not None:
            fa_extra["s_aux"] = self._fa3_sinks
        if self._fa3_window_size != (-1, -1):
            fa_extra["window_size"] = self._fa3_window_size

        if ctx.is_prefill:
            cu_q = ctx.cu_seqlens_q
            cu_k = ctx.cu_seqlens_k
            msq = ctx.max_seqlen_q
            msk = ctx.max_seqlen_k
            bt = ctx.block_tables

            if self.attention_chunk_size is not None:
                cu_q, cu_k, msq, msk, bt = _chunked_prefill_remap(
                    cu_q, cu_k, bt, self.attention_chunk_size, self._block_size,
                )

            if bt is not None:
                return self.prefill_op(
                    q, k_cache, v_cache,
                    cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=msq, max_seqlen_k=msk,
                    softmax_scale=self.scale, causal=True,
                    block_table=bt, **fa_extra,
                )
            return self.prefill_op(
                q, k, v,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=msq, max_seqlen_k=msk,
                softmax_scale=self.scale, causal=True,
                **fa_extra,
            )

        cache_seqlens = ctx.context_lens
        bt = ctx.block_tables
        max_ctx = ctx.max_context_len

        if self.attention_chunk_size is not None:
            cache_seqlens, bt, max_ctx = _chunked_decode_remap(
                cache_seqlens, bt, self.attention_chunk_size, self._block_size,
            )

        return self.decode_op(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens, block_table=bt,
            softmax_scale=self.scale, causal=True,
            max_seq_len=max_ctx, **fa_extra,
        )

    def _forward_mixed(self, q, k_cache, v_cache, ctx):
        fa_extra = {}
        if self._fa3_sinks is not None:
            fa_extra["s_aux"] = self._fa3_sinks
        if self._fa3_window_size != (-1, -1):
            fa_extra["window_size"] = self._fa3_window_size

        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)

        if np_ > 0:
            cu_q = ctx.prefill_cu_seqlens_q
            cu_k = ctx.prefill_cu_seqlens_k
            msq = ctx.prefill_max_seqlen_q
            msk = ctx.prefill_max_seqlen_k
            bt = ctx.prefill_block_tables

            if self.attention_chunk_size is not None:
                cu_q, cu_k, msq, msk, bt = _chunked_prefill_remap(
                    cu_q, cu_k, bt, self.attention_chunk_size, self._block_size,
                )

            pq = q[:np_].contiguous() if self._use_trtllm else q[:np_]
            out[:np_] = self.prefill_op(
                pq, k_cache, v_cache,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=msq, max_seqlen_k=msk,
                softmax_scale=self.scale, causal=True,
                block_table=bt, **fa_extra,
            )

        if nd > 0:
            cache_seqlens = ctx.decode_context_lens
            bt = ctx.decode_block_tables
            max_ctx = ctx.decode_max_context_len

            if self.attention_chunk_size is not None:
                cache_seqlens, bt, max_ctx = _chunked_decode_remap(
                    cache_seqlens, bt,
                    self.attention_chunk_size, self._block_size,
                )

            out[np_:] = self.decode_op(
                q[np_:], k_cache, v_cache,
                cache_seqlens=cache_seqlens, block_table=bt,
                softmax_scale=self.scale, causal=True,
                max_seq_len=max_ctx, **fa_extra,
            )
        return out
