"""TRTLLM-gen paged attention prefill kernel (via FlashInfer, Blackwell only).

Accepts the same cu_seqlens-based interface as FlashAttnPrefill so that
LlamaAttention can dispatch to either backend without branch logic.
"""

import torch
import torch.nn as nn
from flashinfer.prefill import trtllm_batch_context_with_kv_cache


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

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, softmax_scale=None,
                causal=True, block_table=None, window_size=None,
                s_aux=None, **kwargs):
        # ``window_size`` is the FA convention (left, right); ``s_aux`` is the
        # FA3 name for GPT-OSS attention sinks (one extra logit per head with
        # value 0). Both change outputs and must never be silently dropped.
        window_left = int(window_size[0]) if window_size is not None else -1
        if block_table is not None:
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
                bmm1_scale=softmax_scale if softmax_scale is not None else self.sm_scale,
                bmm2_scale=1.0,
                batch_size=batch_size,
                cum_seq_lens_q=cu_seqlens_q,
                cum_seq_lens_kv=cu_seqlens_k,
                window_left=window_left,
                # The TRTLLM kernel requires fp32 sinks (rejects bf16).
                sinks=s_aux.to(torch.float32) if s_aux is not None else None,
                kv_layout="HND",
            )
        from flash_attn import flash_attn_varlen_func
        fa2_kwargs = {}
        if window_size is not None:
            fa2_kwargs["window_size"] = tuple(window_size)
        if s_aux is None:
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale if softmax_scale is not None else self.sm_scale,
                causal=causal, **fa2_kwargs,
            )
        # FA2 has no native sink support. A sink is an extra softmax logit
        # with value 0, so it only rescales the denominator:
        #   out_sink = out * D / (D + exp(s)) = out * sigmoid(lse - s)
        # with lse = log(D) the per-row logsumexp FA2 already computes.
        out, lse, _ = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale if softmax_scale is not None else self.sm_scale,
            causal=causal, return_attn_probs=True, **fa2_kwargs,
        )
        # lse: [num_heads, total_q] fp32 -> scale [total_q, num_heads, 1]
        sink_scale = torch.sigmoid(
            lse.to(torch.float32) - s_aux.to(torch.float32)[:, None]
        ).transpose(0, 1).unsqueeze(-1)
        return (out.to(torch.float32) * sink_scale).to(out.dtype)
