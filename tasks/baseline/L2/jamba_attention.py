"""Jamba's multi-head attention block (no RoPE, no QK-norm).

Reference: ``transformers.models.jamba.modeling_jamba.JambaAttention``
            and ``vllm.model_executor.layers.attention``.

Key difference from :class:`L2.attention.LlamaAttention`: Jamba does NOT
use rotary position embeddings.  Position information enters the network
through Mamba's selective scan (data-dependent recurrence) plus the
ordering of attention layers; no positional embedding is applied to Q
or K.

Cache layout & strategy (mirrors vLLM's Jamba forward path):

  * **Paged KV cache** allocated by the engine and bound to
    ``self.k_cache`` / ``self.v_cache`` (one slice per attention layer).
    Layout depends on the auto-detected backend (TRTLLM-gen on
    Blackwell uses HND ``[num_blocks, num_kv_heads, page_size,
    head_dim]`` with page_size=16; FA3/FA2 elsewhere uses NHD
    ``[num_blocks, page_size, num_kv_heads, head_dim]`` with
    page_size=256).  This matches kb-nano's standard
    ``LlamaEngine.allocate_kv_cache`` pattern.

  * **Prefill**: paged-context attention via ``TRTLLMPrefill`` /
    ``FlashAttnPrefill`` reading from the paged cache (the same
    kernels vLLM uses).  Q/K/V come from the [B, T, h] left-padded
    input; K/V are written to the paged cache via ``StoreKVCache``,
    then attention reads them back through ``block_table`` +
    ``cu_seqlens``.  This matches vLLM's TRTLLM prefill numerics so
    drift vs the reference stays small.

  * **Decode**: paged attention via ``FlashAttnDecode`` /
    ``TRTLLMDecode`` reading the paged caches with ``block_table`` +
    ``cache_seqlens`` from ``get_context()``.  Wasted-tail compute
    (the old dense slab's masked region beyond ``cur_len``) is
    eliminated -- attention only touches valid positions.

  * **Mamba layers** in Jamba consume the standard
    ``mamba_state`` / ``mamba_metadata`` Context fields and are not
    affected by this module.

Forward signature: ``forward(positions, hidden_states)`` where
``hidden_states`` is ``[B, T, hidden]`` (T == 1 in decode, T == max_prompt
in prefill) -- same convention as :class:`L2.attention.LlamaAttention`.

L1 ops used (no ``torch.nn.functional`` / external lib leaks):
  * ``Linear``                       (Q/K/V/O)
  * ``TRTLLMDecode`` / ``FlashAttnDecode``     (paged decode)
  * ``TRTLLMPrefill`` / ``FlashAttnPrefill``    (paged context prefill)
  * ``StoreKVCacheHND`` / ``StoreKVCache``      (paged-cache write)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.context import get_attn_backend_config, get_context
from ..L1.linear import Linear


class JambaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int | None = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.layer_idx = layer_idx

        self.q_proj = Linear(
            hidden_size, num_attention_heads * self.head_dim, bias=False,
        )
        self.k_proj = Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=False,
        )
        self.v_proj = Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=False,
        )
        self.o_proj = Linear(
            num_attention_heads * self.head_dim, hidden_size, bias=False,
        )

        # Paged attention dispatch.  Mirrors ``L2.attention_impl.Attention``:
        # on Blackwell (sm_100+) we use TRTLLM-gen paged kernels via
        # FlashInfer (HND layout, ``block_size=16``); elsewhere we use
        # FA3/FA2 paged kernels (NHD, ``block_size=256``).  Same dispatch
        # for both prefill and decode so numerics match vLLM's choice.
        attn_cfg = get_attn_backend_config()
        self._use_trtllm = attn_cfg.use_trtllm
        self._block_size = attn_cfg.block_size

        if self._use_trtllm:
            from ..L1.flashinfer_decode import TRTLLMDecode
            from ..L1.flashinfer_prefill import TRTLLMPrefill
            from ..L1.store_kvcache import StoreKVCacheHND
            self.decode_attn = TRTLLMDecode(
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
            self.prefill_attn = TRTLLMPrefill(
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
            self.store_kv = StoreKVCacheHND(page_size=attn_cfg.block_size)
        else:
            from ..L1.flash_attn_decode import FlashAttnDecode
            from ..L1.flash_attn_prefill import FlashAttnPrefill
            from ..L1.store_kvcache import StoreKVCache
            self.decode_attn = FlashAttnDecode(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
            self.prefill_attn = FlashAttnPrefill(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
            self.store_kv = StoreKVCache()

        # Per-layer paged caches; bound by the engine after KV allocation.
        # NHD layout: [num_blocks, block_size, num_kv_heads, head_dim]
        # HND layout: [num_blocks, num_kv_heads, block_size, head_dim]
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def forward(
        self,
        positions: torch.Tensor | None,    # unused (no RoPE); kept for
                                            # signature parity with Llama.
        hidden_states: torch.Tensor,        # [B, T, hidden]
    ) -> torch.Tensor:
        """Forward.  Reads per-step paged-KV state from the global
        ``Context`` (populated by ``set_jamba_context``).

        * **Prefill** (``ctx.is_prefill == True``): batched dense
          attention against ``ctx.prefill_attn_mask``.  After computing
          attention, K/V are also written to the paged cache via
          ``ctx.slot_mapping`` so decode steps can read them.

        * **Decode** (``ctx.is_prefill == False``): paged attention via
          ``FlashAttnDecode`` with ``ctx.block_tables`` /
          ``ctx.context_lens``.  K/V for the new step are written to
          the paged cache via ``ctx.slot_mapping`` (one slot per row)
          before the kernel reads them.
        """
        ctx = get_context()
        if ctx.is_prefill:
            return self._forward_prefill(hidden_states, ctx)
        return self._forward_decode(hidden_states, ctx)

    # ------------------------------------------------------------------
    # Prefill: paged-context attention via TRTLLMPrefill / FlashAttnPrefill.
    #
    # The L4 model passes ``hidden_states`` as ``[B, T_max, h]`` (left-
    # padded).  We flatten to varlen format ``[total_real_tokens, h]``
    # using the per-row prompt lengths from ``ctx.cu_seqlens_q``, run
    # paged attention with a side-write to the paged cache, then scatter
    # the output back to ``[B, T_max, h]`` for the downstream layers.
    #
    # This matches vLLM's prefill kernel choice exactly (TRTLLM-gen on
    # Blackwell, FA3 on Hopper), so per-step numerics drift vs the
    # reference is bounded by hardware-level rounding rather than
    # cross-kernel algorithm differences.
    # ------------------------------------------------------------------
    def _forward_prefill(self, hidden_states: torch.Tensor, ctx) -> torch.Tensor:
        b, t_max, h = hidden_states.shape

        # ``ctx.cu_seqlens_q`` describes the left-padded → flat-varlen
        # remap: for row i, the real tokens occupy positions
        # ``[t_max - prompt_lens[i], t_max)`` in the [B, T_max] grid.
        # ``ctx.flat_to_grid`` maps each real-token slot k in the flat
        # tensor back to the matching index in the dense [B*T_max] view
        # so we can gather/scatter without a host sync.
        cu_q = ctx.cu_seqlens_q
        flat_idx = ctx.flat_to_grid
        assert cu_q is not None and flat_idx is not None, (
            "JambaAttention prefill requires cu_seqlens_q and "
            "flat_to_grid on the Context.  The engine populates these "
            "in set_jamba_context."
        )

        # Project Q, K, V on the dense [B*T_max, h] layout, then
        # gather only the real-token rows.
        flat_dense = hidden_states.reshape(b * t_max, h)
        flat_real = flat_dense.index_select(0, flat_idx)  # [N, h]

        q = self.q_proj(flat_real).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(flat_real).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(flat_real).view(-1, self.num_kv_heads, self.head_dim)

        # Write K, V to paged cache.  ``slot_mapping`` here is the flat
        # version (one entry per real token) -- the engine builds it
        # from the same flat_idx + per-row block_table.
        self.store_kv(k, v, self.k_cache, self.v_cache, ctx.slot_mapping)

        # Paged-context attention.  TRTLLMPrefill / FlashAttnPrefill read
        # K, V from the paged cache (block_table) -- the just-written
        # K/V are visible because ``store_kv`` ran on the same stream.
        max_seqlen = ctx.max_seqlen_q
        out = self.prefill_attn(
            q, self.k_cache, self.v_cache,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=ctx.cu_seqlens_k,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=ctx.max_seqlen_k,
            softmax_scale=self.scaling,
            causal=True,
            block_table=ctx.block_tables,
        )
        # ``out`` is [N, num_heads, head_dim].  Project + scatter back
        # to [B, T_max, h].  Padded slots stay 0; downstream layers'
        # subsequent residual + RMSNorm passes operate on a tensor
        # whose padded rows happen to be zero, which is benign for
        # the last-token logit extraction the engine does after the
        # forward.
        out = out.reshape(-1, self.num_heads * self.head_dim)
        out = self.o_proj(out)  # [N, h]
        # Scatter back into the full [B*T_max, h] layout.
        scattered = torch.zeros(
            b * t_max, h, dtype=out.dtype, device=out.device,
        )
        scattered.index_copy_(0, flat_idx, out)
        return scattered.reshape(b, t_max, h)

    # ------------------------------------------------------------------
    # Decode: paged attention via FlashAttnDecode + per-step KV write.
    # ------------------------------------------------------------------
    def _forward_decode(self, hidden_states: torch.Tensor, ctx) -> torch.Tensor:
        # ``hidden_states`` is [B, 1, hidden] for decode; flatten to [B, hidden].
        b, t, _ = hidden_states.shape
        assert t == 1, f"JambaAttention decode expects T=1, got T={t}"
        h = hidden_states.view(b, -1)

        q = self.q_proj(h).view(b, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(b, self.num_kv_heads, self.head_dim)
        v = self.v_proj(h).view(b, self.num_kv_heads, self.head_dim)

        # Write the new step's K, V into the paged cache at the slots
        # the engine has already computed for this step.
        self.store_kv(k, v, self.k_cache, self.v_cache, ctx.slot_mapping)

        # Paged decode attention.  FlashAttnDecode wraps FA3
        # ``flash_attn_varlen_func`` (or FA2 ``flash_attn_with_kvcache``)
        # with the paged cache + block_table + cache_seqlens read from
        # the Context.  The kernel only touches valid positions
        # (``[:context_lens[i]]`` per batch row), eliminating the
        # masked-tail compute the previous dense-slab path paid.
        out = self.decode_attn(
            q, self.k_cache, self.v_cache,
            cache_seqlens=ctx.context_lens,
            block_table=ctx.block_tables,
            softmax_scale=self.scaling,
            causal=True,
            max_seq_len=ctx.max_context_len,
        )
        # FA returns [B, num_heads, head_dim] (FA3 path) or already
        # squeezed (FA2 fallback in FlashAttnDecode); either way reshape
        # to [B, 1, hidden] for o_proj.
        out = out.contiguous().view(b, 1, self.num_heads * self.head_dim)
        return self.o_proj(out)
