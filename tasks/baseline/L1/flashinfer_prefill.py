"""TRTLLM-gen paged attention prefill kernel (via FlashInfer, Blackwell only)."""

import torch
import torch.nn as nn
from flashinfer.prefill import trtllm_batch_context_with_kv_cache


class TRTLLMPrefill(nn.Module):
    """Wraps trtllm_batch_context_with_kv_cache for paged KV cache prefill.

    Takes block_tables, seq_lens, and cumulative sequence lengths directly.
    """
    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        self._workspace = torch.zeros(
            512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )

    def forward(self, q, k_cache, v_cache, block_tables, seq_lens,
                max_q_len, max_kv_len, batch_size,
                cum_seq_lens_q, cum_seq_lens_kv, **kwargs):
        return trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self._workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=self.sm_scale,
            bmm2_scale=1.0,
            batch_size=batch_size,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            kv_layout="NHD",
        )
