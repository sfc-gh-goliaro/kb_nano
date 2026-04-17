"""Gated Delta Net (GDN) recurrence (L1).

Wraps the two external kernels Qwen3-Next's linear-attention path needs:

  * Prefill chunked path  -> ``flashinfer.gdn_prefill.chunk_gated_delta_rule``
  * Decode recurrent path -> vLLM's bundled FLA
    ``fused_recurrent_gated_delta_rule``

Both are exposed as ``nn.Module`` subclasses so L2 callers can compose
them without importing FlashInfer or vLLM directly.

Side-effect on import: configures Triton's allocator to a CUDA int8 buffer.
This matches how the original L2 module bootstrapped the kernels and is
required before invoking either Triton-backed function.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Triton allocator setup is a hard prerequisite for the FlashInfer / vLLM
# Triton kernels below. Centralizing it here means individual L2 callers
# don't have to know about it.
import triton as _triton

_triton.set_allocator(
    lambda size, alignment, stream: torch.empty(
        size, device="cuda", dtype=torch.int8
    )
)

from flashinfer.gdn_prefill import (
    chunk_gated_delta_rule as _fi_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule as _vllm_fused_recurrent,
)


class GDNChunkPrefill(nn.Module):
    """FlashInfer chunked GDN prefill (Hopper, SM90+).

    Expects q/k/v with the leading batch dim already squeezed away
    (FlashInfer convention). ``g`` should be in linear (already exp'd)
    space and ``beta`` / ``initial_state`` should be float32.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor,
    ):
        return _fi_chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )


class GDNFusedRecurrent(nn.Module):
    """vLLM-bundled FLA fused recurrent GDN kernel (decode path).

    Keeps ``ssm_state`` in caller-provided dtype (typically bf16) for
    bitwise alignment with vLLM's runtime.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        inplace_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        return _vllm_fused_recurrent(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
