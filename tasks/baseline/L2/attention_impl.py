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
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....infra.context import get_context, get_attn_backend_config
from ..L1.store_kvcache import StoreKVCache, StoreKVCacheHND


class Attention(nn.Module):

    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int | None = None,
                 sliding_window: int | None = None,
                 sinks: torch.nn.Parameter | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window
        self.sinks = sinks

        self.k_cache = self.v_cache = torch.tensor([])

        attn_cfg = get_attn_backend_config()
        self._use_trtllm = attn_cfg.use_trtllm
        self._use_sdpa = sinks is not None

        if self._use_sdpa:
            self.store_kvcache = StoreKVCache()
        elif self._use_trtllm:
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

    def set_trtllm_workspace(self, workspace: torch.Tensor):
        if self._use_trtllm:
            self.decode_op._workspace = workspace
            self.prefill_op._workspace = workspace

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        N = query.shape[0]

        q = query.view(N, self.num_heads, self.head_size)
        k = key.view(N, self.num_kv_heads, self.head_size)
        v = value.view(N, self.num_kv_heads, self.head_size)

        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.store_kvcache(k, v, k_cache, v_cache, ctx.slot_mapping)

        if self._use_sdpa:
            o = self._forward_sdpa(q, k, v, k_cache, v_cache, ctx)
        elif ctx.is_mixed:
            o = self._forward_mixed(q, k_cache, v_cache, ctx)
        else:
            o = self._forward_pure(q, k, v, k_cache, v_cache, ctx)

        return o.reshape(N, self.num_heads * self.head_size)

    def _get_window_size(self):
        if self.sliding_window is not None:
            return (self.sliding_window, 0)
        return (-1, -1)

    def _forward_sdpa(self, q, k, v, k_cache, v_cache, ctx):
        """SDPA-based forward path for attention with sinks and/or sliding window."""
        if ctx.is_prefill:
            return self._forward_sdpa_prefill(q, k, v, ctx)
        return self._forward_sdpa_decode(q, k_cache, v_cache, ctx)

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

    def _forward_pure(self, q, k, v, k_cache, v_cache, ctx):
        sw_kw = {}
        if self.sliding_window is not None:
            sw_kw["window_size"] = self._get_window_size()
        if ctx.is_prefill:
            if ctx.block_tables is not None:
                return self.prefill_op(
                    q, k_cache, v_cache,
                    cu_seqlens_q=ctx.cu_seqlens_q,
                    cu_seqlens_k=ctx.cu_seqlens_k,
                    max_seqlen_q=ctx.max_seqlen_q,
                    max_seqlen_k=ctx.max_seqlen_k,
                    softmax_scale=self.scale, causal=True,
                    block_table=ctx.block_tables,
                    **sw_kw,
                )
            return self.prefill_op(
                q, k, v,
                cu_seqlens_q=ctx.cu_seqlens_q, cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q, max_seqlen_k=ctx.max_seqlen_k,
                softmax_scale=self.scale, causal=True,
                **sw_kw,
            )
        return self.decode_op(
            q, k_cache, v_cache,
            cache_seqlens=ctx.context_lens, block_table=ctx.block_tables,
            softmax_scale=self.scale, causal=True,
            max_seq_len=ctx.max_context_len,
            **sw_kw,
        )

    def _forward_mixed(self, q, k_cache, v_cache, ctx):
        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)

        if np_ > 0:
            pq = q[:np_].contiguous() if self._use_trtllm else q[:np_]
            out[:np_] = self.prefill_op(
                pq, k_cache, v_cache,
                cu_seqlens_q=ctx.prefill_cu_seqlens_q,
                cu_seqlens_k=ctx.prefill_cu_seqlens_k,
                max_seqlen_q=ctx.prefill_max_seqlen_q,
                max_seqlen_k=ctx.prefill_max_seqlen_k,
                softmax_scale=self.scale, causal=True,
                block_table=ctx.prefill_block_tables,
            )

        if nd > 0:
            out[np_:] = self.decode_op(
                q[np_:], k_cache, v_cache,
                cache_seqlens=ctx.decode_context_lens,
                block_table=ctx.decode_block_tables,
                softmax_scale=self.scale, causal=True,
                max_seq_len=ctx.decode_max_context_len,
            )
        return out
