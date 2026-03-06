"""FlashInfer paged attention prefill kernel."""

import torch
import torch.nn as nn
from flashinfer import BatchPrefillWithPagedKVCacheWrapper


class FlashInferPrefill(nn.Module):
    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 page_size: int):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self._workspace = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8,
                                      device="cuda")
        self._wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self._workspace, kv_layout="NHD",
        )

    def plan(self, qo_indptr: torch.Tensor, paged_kv_indptr: torch.Tensor,
             paged_kv_indices: torch.Tensor,
             paged_kv_last_page_len: torch.Tensor,
             q_dtype: torch.dtype):
        self._wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=self.page_size,
            causal=True,
            sm_scale=self.head_dim ** -0.5,
            q_data_type=q_dtype,
        )

    def forward(self, q, k_cache, v_cache, **kwargs):
        return self._wrapper.run(q, (k_cache, v_cache))
