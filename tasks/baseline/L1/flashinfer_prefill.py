"""TRTLLM-gen paged attention prefill kernel (via FlashInfer, Blackwell only).

Accepts the same cu_seqlens-based interface as FlashAttnPrefill so that
LlamaAttention can dispatch to either backend without branch logic.

The TRTLLM-gen context kernel only implements bottom-right-aligned causal
masking, so non-causal paged prefill (Whisper-style cross-attention, where
every decoder query attends to the whole encoder sequence) is served by
FlashInfer's ``BatchPrefillWithPagedKVCacheWrapper`` instead.

``Attention`` hands sliding-window and attention-sink parameters down as the
``window_size``/``s_aux`` kwargs that vLLM's FlashAttention uses; both are
translated here to the TRTLLM kernels' ``window_left``/``sinks``.
"""

import torch
import torch.nn as nn
from flashinfer.prefill import trtllm_batch_context_with_kv_cache

from ._fa_backend import fa_version_for_head_dim as _fa_version_for_head_dim
from ._fa_backend import VLLM_FA_AVAILABLE as _VLLM_FA_AVAILABLE
from ._fa_backend import vllm_fa_varlen_func as _vllm_fa_varlen_func


def _window_left(window_size) -> int:
    """vLLM-style ``(left, right)`` window -> TRTLLM's ``window_left``."""
    if window_size is None:
        return -1
    return int(window_size[0])


def as_sinks(s_aux):
    """TRTLLM's sink kernels require fp32, matching vLLM's FlashInfer backend."""
    if s_aux is None:
        return None
    return s_aux if s_aux.dtype == torch.float32 else s_aux.to(torch.float32)


# One wrapper (and one workspace) per (device, layout) shared by every layer:
# ``plan()`` and ``run()`` are always called back-to-back, and a per-layer
# wrapper would otherwise pin 128 MiB apiece across a model's decoder stack.
_NONCAUSAL_WRAPPERS: dict[tuple, object] = {}


def _get_noncausal_wrapper(device: torch.device):
    key = (device.type, device.index)
    wrapper = _NONCAUSAL_WRAPPERS.get(key)
    if wrapper is None:
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            kv_layout="HND",
        )
        _NONCAUSAL_WRAPPERS[key] = wrapper
    return wrapper


class TRTLLMPrefill(nn.Module):
    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 workspace: torch.Tensor | None = None):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        if workspace is None:
            workspace = torch.zeros(
                512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
        self._workspace = workspace
        self._noncausal_wrapper = None

    def _forward_noncausal_paged(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                                 block_table, sm_scale, window_left=-1):
        """Non-causal paged prefill via FlashInfer's batch-prefill wrapper.

        ``k``/``v`` are HND paged caches: [num_blocks, num_kv_heads, page_size,
        head_dim].  The wrapper wants CSR-style page indices rather than a
        dense block table, so derive them from the per-request KV lengths.
        """
        page_size = k.shape[2]
        seq_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        num_pages = (seq_lens + page_size - 1) // page_size
        kv_indptr = torch.zeros(
            seq_lens.numel() + 1, dtype=torch.int32, device=q.device,
        )
        torch.cumsum(num_pages, dim=0, out=kv_indptr[1:])
        page_slot = torch.arange(block_table.shape[1], device=q.device)
        kv_indices = block_table[page_slot.unsqueeze(0) < num_pages.unsqueeze(1)]
        # A KV length that is an exact multiple of page_size fills its last page.
        last_page_len = seq_lens - (num_pages - 1) * page_size

        if self._noncausal_wrapper is None:
            self._noncausal_wrapper = _get_noncausal_wrapper(q.device)
        self._noncausal_wrapper.plan(
            cu_seqlens_q.to(torch.int32),
            kv_indptr,
            kv_indices.to(torch.int32),
            last_page_len.to(torch.int32),
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            page_size,
            causal=False,
            sm_scale=sm_scale,
            window_left=window_left,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
        )
        return self._noncausal_wrapper.run(q.contiguous(), (k, v))

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, softmax_scale=None,
                causal=True, block_table=None, s_aux=None, window_size=None,
                **kwargs):
        sm_scale = softmax_scale if softmax_scale is not None else self.sm_scale
        window_left = _window_left(window_size)
        if block_table is not None:
            if not causal:
                return self._forward_noncausal_paged(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, block_table, sm_scale,
                    window_left,
                )
            q = q.contiguous()
            seq_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            batch_size = seq_lens.shape[0]
            return trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=(k, v),
                workspace_buffer=self._workspace,
                block_tables=block_table,
                seq_lens=seq_lens,
                max_q_len=max_seqlen_q,
                max_kv_len=max_seqlen_k,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
                batch_size=batch_size,
                cum_seq_lens_q=cu_seqlens_q,
                cum_seq_lens_kv=cu_seqlens_k,
                window_left=window_left,
                sinks=as_sinks(s_aux),
                kv_layout="HND",
            )
        # No paged cache (profile/warmup runs): vLLM's FA understands both sinks
        # and the sliding window, so prefer it over upstream flash_attn. Resolve
        # the FA version from the head size -- FA4's Blackwell kernels exceed the
        # sm100 TMEM budget above head_dim 128 (Gemma-4 runs at 256) and assert
        # inside flash_fwd_sm100.
        if _VLLM_FA_AVAILABLE:
            return _vllm_fa_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                softmax_scale=sm_scale, causal=causal,
                fa_version=_fa_version_for_head_dim(q.shape[-1]),
                **({"s_aux": s_aux} if s_aux is not None else {}),
                **({"window_size": list(window_size)} if window_size is not None else {}),
            )
        from flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            softmax_scale=sm_scale,
            causal=causal,
        )
