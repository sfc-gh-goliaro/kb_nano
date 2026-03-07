"""TRTLLM-gen paged attention decode kernel (via FlashInfer, Blackwell only)."""

import torch
import torch.nn as nn
from flashinfer.decode import trtllm_batch_decode_with_kv_cache


class TRTLLMDecode(nn.Module):
    """Wraps trtllm_batch_decode_with_kv_cache for paged KV cache decode.

    Interface mirrors flash_attn_with_kvcache: takes block_tables and
    cache_seqlens as GPU tensors, fully CUDA-graph-compatible.
    """
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

    def forward(self, q, k_cache, v_cache, cache_seqlens, block_table,
                max_seq_len, **kwargs):
        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self._workspace,
            block_tables=block_table,
            seq_lens=cache_seqlens,
            max_seq_len=max_seq_len,
            bmm1_scale=self.sm_scale,
            bmm2_scale=1.0,
            kv_layout="HND",
        )
