"""Flash attention decode kernel (with paged KV cache)."""

import torch.nn as nn
from flash_attn import flash_attn_with_kvcache


class FlashAttnDecode(nn.Module):
    def forward(self, q, k_cache, v_cache, cache_seqlens=None, **kwargs):
        return flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens, **kwargs)
