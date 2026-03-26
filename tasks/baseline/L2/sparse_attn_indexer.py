"""DSA (DeepSeek Sparse Attention) indexer.

Produces top-k token indices for sparse attention. Components:
- wq_b: replicated linear (q_lora_rank -> head_dim * n_head) with FP8
- wk: replicated linear (hidden_size -> head_dim) with FP8
- k_norm: LayerNorm(head_dim, eps=1e-6)
- weights_proj: replicated linear (hidden_size -> n_head) — NO FP8
- Own K cache: [num_blocks, block_size, 132] uint8
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.context import get_context
from ..L1.layer_norm import LayerNorm
from ..L1.fp8_linear import _per_token_group_quant_fp8
from ..L1.indexer_k_cache import IndexerKCacheStore, IndexerKCacheGather
from ..L1.fp8_mqa_logits import Fp8MQALogits, Fp8PagedMQALogitsMetadata
from ..L1.top_k_per_row import TopKPerRow


class ReplicatedLinear(nn.Module):
    """Linear layer replicated across all TP ranks (no sharding)."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None):
        super().__init__()
        from ..L1.fp8_linear import Fp8Linear
        from ..L1.linear import Linear
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            import math
            _FP8_BLOCK = 128
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(math.ceil(output_size / _FP8_BLOCK),
                              math.ceil(input_size / _FP8_BLOCK),
                              dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)
            self.weight_scale_inv.weight_loader = lambda p, w: p.data.copy_(w)
            self.linear_op = Fp8Linear()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)
            self.linear_op = Linear()

        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return self.linear_op(x, self.weight, self.bias)


class SparseAttnIndexer(nn.Module):
    """DSA sparse attention indexer.

    Produces topk_indices [M, topk_tokens] consumed by sparse FlashMLA.

    Args:
        hidden_size: model hidden dimension
        q_lora_rank: query latent dimension
        n_head: number of indexer heads (64)
        head_dim: indexer head dimension (128)
        rope_dim: RoPE dimension (64)
        topk_tokens: number of tokens to select per query
        quant_config: FP8 quantization config
    """

    def __init__(self, hidden_size: int, q_lora_rank: int,
                 n_head: int, head_dim: int, rope_dim: int,
                 topk_tokens: int, quant_config: dict | None = None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.topk_tokens = topk_tokens
        self.q_lora_rank = q_lora_rank
        self.softmax_scale = head_dim ** -0.5

        self.wq_b = ReplicatedLinear(q_lora_rank, head_dim * n_head,
                                     quant_config=quant_config)
        self.wk = ReplicatedLinear(hidden_size, head_dim,
                                   quant_config=quant_config)
        self.k_norm = LayerNorm(head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(hidden_size, n_head)  # NO FP8

        self.k_cache_store = IndexerKCacheStore()
        self.k_cache_gather = IndexerKCacheGather()
        self.fp8_mqa_logits = Fp8MQALogits()
        self.paged_mqa_metadata = Fp8PagedMQALogitsMetadata()
        self.topk_per_row = TopKPerRow()

        # Indexer K cache: [num_blocks, block_size, 132] uint8
        # Will be allocated by engine and set externally
        self.indexer_k_cache = torch.tensor([])

        self._quant_block_size = 128

    def forward(self, hidden_states: torch.Tensor, q_latent: torch.Tensor,
                positions: torch.Tensor, rope_emb: nn.Module) -> torch.Tensor:
        """
        Args:
            hidden_states: [M, hidden_size]
            q_latent: [M, q_lora_rank] - compressed query from fused_qkv_a_proj
            positions: [M] position ids
            rope_emb: YarnRotaryEmbedding for indexer

        Returns:
            topk_indices: [M, topk_tokens] int32
        """
        ctx = get_context()
        M = hidden_states.shape[0]

        # Q path: wq_b -> reshape -> split pe/nope -> RoPE -> concat
        q = self.wq_b(q_latent)  # [M, head_dim * n_head]
        q = q.view(M, self.n_head, self.head_dim)
        q_pe = q[..., :self.rope_dim]  # [M, n_head, rope_dim]
        q_nope = q[..., self.rope_dim:]  # [M, n_head, head_dim - rope_dim]

        # K path: wk -> k_norm -> split pe/nope
        k = self.wk(hidden_states)  # [M, head_dim]
        k = self.k_norm(k)
        k_pe = k[:, :self.rope_dim]  # [M, rope_dim]
        k_nope = k[:, self.rope_dim:]  # [M, head_dim - rope_dim]

        # RoPE on pe components
        q_pe, k_pe_out = rope_emb(positions, q_pe, k_pe.unsqueeze(1))
        q_pe = q_pe.reshape(M, self.n_head, self.rope_dim)
        k_pe_out = k_pe_out.reshape(M, 1, self.rope_dim)

        # Concat pe + nope
        q = torch.cat([q_pe, q_nope], dim=-1)  # [M, n_head, head_dim]
        k = torch.cat([k_pe_out.squeeze(1), k_nope], dim=-1)  # [M, head_dim]

        # FP8 quantize Q
        q_flat = q.reshape(-1, self.head_dim)
        q_fp8 = torch.empty_like(q_flat, dtype=torch.float8_e4m3fn)
        q_scale = torch.empty(q_flat.shape[0], self.head_dim // self._quant_block_size,
                              dtype=torch.float32, device=q_flat.device)
        _per_token_group_quant_fp8(q_flat, q_fp8, q_scale)
        q_fp8 = q_fp8.view(M, self.n_head, self.head_dim)
        q_scale = q_scale.view(M, self.n_head, -1)

        # Store K to indexer cache
        if ctx.slot_mapping is not None and self.indexer_k_cache.numel():
            self.k_cache_store(k, self.indexer_k_cache, ctx.slot_mapping)

        weights = self.weights_proj(hidden_states)  # [M, n_head]
        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head ** -0.5
        )
        weights = weights.squeeze(-1)

        # Compute logits and top-k
        topk_indices = torch.full((M, self.topk_tokens), -1, dtype=torch.int32,
                                  device=hidden_states.device)

        if ctx.is_prefill or (ctx.is_mixed and ctx.num_prefill_tokens > 0):
            # Prefill path: gather K from cache, compute logits, top-k
            if ctx.is_mixed:
                np_ = ctx.num_prefill_tokens
                cu_q = ctx.prefill_cu_seqlens_q
                cu_k = ctx.prefill_cu_seqlens_k
                bt = ctx.prefill_block_tables
                q_fp8_pf = q_fp8[:np_]
                weights_pf = weights[:np_]
            else:
                np_ = M
                cu_q = ctx.cu_seqlens_q
                cu_k = ctx.cu_seqlens_k
                bt = ctx.block_tables
                q_fp8_pf = q_fp8
                weights_pf = weights

            if cu_q is None or cu_k is None or bt is None:
                return topk_indices

            k_fp8, k_scale_flat = self.k_cache_gather(
                self.indexer_k_cache, bt, cu_k)

            logits = self.fp8_mqa_logits.forward_prefill(
                q_fp8_pf.view(-1, self.n_head, self.head_dim),
                (k_fp8, k_scale_flat),
                weights_pf,
                cu_q[:-1].int(),
                cu_k[1:].int(),
            )

            topk_indices[:np_] = self.topk_per_row.forward_prefill(
                logits, cu_q[:-1].int(), cu_k[1:].int(),
                self.topk_tokens,
            )

            if ctx.is_mixed and ctx.num_decode_tokens > 0:
                nd = ctx.num_decode_tokens
                q_fp8_dc = q_fp8[np_:]
                weights_dc = weights[np_:]
                topk_indices[np_:] = self._decode_topk(
                    q_fp8_dc, weights_dc, ctx, hidden_states.device,
                )
        else:
            # Decode path (decode-only batch)
            topk_indices = self._decode_topk(q_fp8, weights, ctx, hidden_states.device)

        return topk_indices

    def _decode_topk(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        ctx,
        device: torch.device,
    ) -> torch.Tensor:
        """Paged FP8 MQA logits + top-k for decode."""
        M = q_fp8.shape[0]
        out = torch.full((M, self.topk_tokens), -1, dtype=torch.int32, device=device)

        if not self.indexer_k_cache.numel():
            return out
        if ctx.decode_context_lens is None or ctx.decode_block_tables is None:
            return out

        block_size = int(self.indexer_k_cache.shape[1])
        max_ctx = int(ctx.decode_max_context_len or ctx.max_context_len or 1)

        if device.type == "cuda":
            num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        else:
            num_sms = 1

        schedule = self.paged_mqa_metadata(
            ctx.decode_context_lens,
            block_size,
            num_sms,
        )
        logits = self.fp8_mqa_logits.forward_decode(
            q_fp8,
            self.indexer_k_cache,
            weights,
            ctx.decode_context_lens,
            ctx.decode_block_tables,
            schedule,
            max_model_len=max_ctx,
        )
        return self.topk_per_row.forward_decode(
            logits, ctx.decode_context_lens, next_n=1, topk=self.topk_tokens,
        )
