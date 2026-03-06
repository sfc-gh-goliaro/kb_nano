"""FlashInfer paged attention decode kernel."""

import torch
import torch.nn as nn
from flashinfer import BatchDecodeWithPagedKVCacheWrapper


class FlashInferDecode(nn.Module):
    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 page_size: int):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self._workspace = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8,
                                      device="cuda")
        self._wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self._workspace, kv_layout="NHD", use_tensor_cores=True,
        )
        self._planned = False

    def plan(self, indptr: torch.Tensor, indices: torch.Tensor,
             last_page_len: torch.Tensor, q_dtype: torch.dtype):
        self._wrapper.plan(
            indptr=indptr, indices=indices, last_page_len=last_page_len,
            num_qo_heads=self.num_qo_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, page_size=self.page_size,
            pos_encoding_mode="NONE",
            sm_scale=self.head_dim ** -0.5,
            q_data_type=q_dtype,
        )
        self._planned = True

    def forward(self, q, k_cache, v_cache, **kwargs):
        return self._wrapper.run(q, (k_cache, v_cache))
