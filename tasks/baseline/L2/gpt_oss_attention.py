"""GPT-OSS attention: GQA with bias, YaRN RoPE, sliding window, and attention sinks.

Attention sinks are virtual attention drains: they add exp(sink) to the
softmax denominator without contributing any value to the output.
Implemented by appending a zero-key / zero-value "virtual" position whose
attention bias equals the per-head sink scalar.

Uses SDPA for the core attention computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....infra.context import get_context
from ....infra.tp import _tp_size, _tp_rank
from .parallel_linear import QKVParallelLinear, RowParallelLinear
from ..L1.store_kvcache import StoreKVCache


class GptOssAttention(nn.Module):
    """GQA attention with bias, sliding window, and attention sinks."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int,
        sliding_window: int | None = None,
    ):
        super().__init__()
        tp = _tp_size()
        self.num_heads = num_attention_heads // tp
        self.num_kv_heads = num_key_value_heads // tp
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5
        self.layer_idx = layer_idx

        # Sliding window on even layers only
        self.sliding_window = sliding_window if layer_idx % 2 == 0 else None

        # QKV with bias
        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads, num_key_value_heads,
            bias=True,
        )
        # O with bias
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
            bias=True,
        )

        # Attention sinks: per-head scalar that acts as a virtual attention drain.
        # Adds exp(sink) to softmax denominator without contributing to output.
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))
        self.sinks.weight_loader = self._sinks_weight_loader

        self.k_cache = self.v_cache = torch.tensor([])
        self.store_kvcache = StoreKVCache()

    def _sinks_weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        heads_per_rank = param.data.size(0)
        start = rank * heads_per_rank
        param.data.copy_(loaded_weight.narrow(0, start, heads_per_rank))

    def _make_attn_bias_with_sinks(self, T: int, S: int, device: torch.device,
                                    dtype: torch.dtype) -> torch.Tensor:
        """Build attention bias: causal + sliding window + virtual sink column.

        Returns bias of shape [H, T, S+1] where the last column is the sink.
        """
        # Start with causal mask [T, S]
        bias = torch.zeros(T, S, device=device, dtype=dtype)
        if T > 1:
            causal = torch.triu(
                torch.full((T, S), float("-inf"), device=device, dtype=dtype),
                diagonal=S - T + 1,
            )
            bias = bias + causal

        # Sliding window mask
        if self.sliding_window is not None and S > self.sliding_window:
            sw_mask = torch.tril(
                torch.full((T, S), float("-inf"), device=device, dtype=dtype),
                diagonal=S - T - self.sliding_window,
            )
            bias = bias + sw_mask

        # Expand to per-head: [H, T, S]
        bias = bias.unsqueeze(0).expand(self.num_heads, -1, -1).contiguous()

        # Append sink column: [H, T, 1]
        # The virtual position has Q @ 0^T = 0, so SDPA score = 0 * scale + sink = sink
        sink_col = self.sinks.view(self.num_heads, 1, 1).expand(-1, T, 1)
        bias = torch.cat([bias, sink_col], dim=-1)  # [H, T, S+1]

        return bias

    def _gather_kv_from_cache(self, k_cache, v_cache, block_tables, context_lens):
        """Gather KV from paged cache into contiguous tensors."""
        B = block_tables.shape[0]
        max_seq = int(context_lens.max().item())
        block_size = k_cache.shape[1]

        k_out = torch.zeros(B, max_seq, self.num_kv_heads, self.head_dim,
                            device=k_cache.device, dtype=k_cache.dtype)
        v_out = torch.zeros_like(k_out)

        for b in range(B):
            seq_len = int(context_lens[b].item())
            num_blocks_needed = (seq_len + block_size - 1) // block_size
            for blk_idx in range(num_blocks_needed):
                block_id = block_tables[b, blk_idx]
                start = blk_idx * block_size
                end = min(start + block_size, seq_len)
                length = end - start
                k_out[b, start:end] = k_cache[block_id, :length]
                v_out[b, start:end] = v_cache[block_id, :length]

        return k_out, v_out

    def forward(self, positions, hidden_states, rotary_emb):
        ctx = get_context()
        N = hidden_states.shape[0]

        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        # YaRN RoPE
        q, k = rotary_emb(positions, q, k)

        # Store KV cache
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.store_kvcache(k, v, k_cache, v_cache, ctx.slot_mapping)

        if ctx.is_prefill:
            cu_q = ctx.cu_seqlens_q
            B = cu_q.shape[0] - 1

            outputs = []
            for b in range(B):
                start = int(cu_q[b].item())
                end = int(cu_q[b + 1].item())
                T = end - start

                q_b = q[start:end]  # [T, H, D]
                k_b = k[start:end]  # [T, Hkv, D]
                v_b = v[start:end]  # [T, Hkv, D]

                # GQA expansion
                if self.num_kv_heads != self.num_heads:
                    rep = self.num_heads // self.num_kv_heads
                    k_b = k_b.repeat_interleave(rep, dim=1)
                    v_b = v_b.repeat_interleave(rep, dim=1)

                # Append virtual zero key/value for sink
                zero_kv = torch.zeros(1, self.num_heads, self.head_dim,
                                      device=q.device, dtype=q.dtype)
                k_ext = torch.cat([k_b, zero_kv], dim=0)  # [T+1, H, D]
                v_ext = torch.cat([v_b, zero_kv], dim=0)  # [T+1, H, D]

                # Reshape to [1, H, T, D] / [1, H, T+1, D]
                q_4d = q_b.transpose(0, 1).unsqueeze(0)
                k_4d = k_ext.transpose(0, 1).unsqueeze(0)
                v_4d = v_ext.transpose(0, 1).unsqueeze(0)

                # Build attention bias with sinks as virtual column
                attn_bias = self._make_attn_bias_with_sinks(
                    T, T, q.device, q.dtype,
                )  # [H, T, T+1]
                attn_bias = attn_bias.unsqueeze(0)  # [1, H, T, T+1]

                o_b = F.scaled_dot_product_attention(
                    q_4d, k_4d, v_4d,
                    attn_mask=attn_bias,
                    scale=self.scaling,
                )  # [1, H, T, D]
                outputs.append(o_b.squeeze(0).transpose(0, 1))  # [T, H, D]

            o = torch.cat(outputs, dim=0)  # [N, H, D]
        else:
            # Decode: gather KV from paged cache
            k_gathered, v_gathered = self._gather_kv_from_cache(
                k_cache, v_cache, ctx.block_tables, ctx.context_lens,
            )
            B = ctx.block_tables.shape[0]
            max_seq = k_gathered.shape[1]

            q_4d = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k_4d = k_gathered.transpose(1, 2)  # [B, Hkv, S, D]
            v_4d = v_gathered.transpose(1, 2)  # [B, Hkv, S, D]

            # GQA expansion
            if self.num_kv_heads != self.num_heads:
                rep = self.num_heads // self.num_kv_heads
                k_4d = k_4d.repeat_interleave(rep, dim=1)
                v_4d = v_4d.repeat_interleave(rep, dim=1)

            # Append virtual zero key/value for sink
            zero_kv = torch.zeros(B, self.num_heads, 1, self.head_dim,
                                  device=q.device, dtype=q.dtype)
            k_4d = torch.cat([k_4d, zero_kv], dim=2)  # [B, H, S+1, D]
            v_4d = torch.cat([v_4d, zero_kv], dim=2)  # [B, H, S+1, D]

            # Build attention bias for decode: [H, 1, S+1]
            attn_bias = torch.zeros(
                self.num_heads, 1, max_seq + 1,
                device=q.device, dtype=q.dtype,
            )

            # Sink column (last position = virtual)
            attn_bias[:, :, -1] = self.sinks.unsqueeze(1)

            # Sliding window: mask out positions beyond window
            if self.sliding_window is not None:
                for b_idx in range(B):
                    seq_len = int(ctx.context_lens[b_idx].item())
                    if seq_len > self.sliding_window:
                        cutoff = seq_len - self.sliding_window
                        if cutoff > 0:
                            attn_bias[:, :, :cutoff] = float("-inf")

            # Mask padding positions (but not the virtual sink at the end)
            for b_idx in range(B):
                seq_len = int(ctx.context_lens[b_idx].item())
                if seq_len < max_seq:
                    attn_bias[:, :, seq_len:max_seq] = float("-inf")

            attn_bias = attn_bias.unsqueeze(0).expand(B, -1, -1, -1)

            o = F.scaled_dot_product_attention(
                q_4d, k_4d, v_4d,
                attn_mask=attn_bias,
                scale=self.scaling,
            )  # [B, H, 1, D]
            o = o.transpose(1, 2).reshape(B, self.num_heads * self.head_dim)
            return self.o_proj(o)

        return self.o_proj(o.reshape(N, self.num_heads * self.head_dim))
