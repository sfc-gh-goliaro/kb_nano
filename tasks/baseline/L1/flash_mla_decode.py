"""FlashMLA decode kernel for MLA (Multi-head Latent Attention)."""

from __future__ import annotations

import torch
import torch.nn as nn

from flash_mla import flash_mla_with_kvcache, get_mla_metadata


class FlashMLADecode(nn.Module):
    """Wraps flash_mla.flash_mla_with_kvcache for paged MLA decode.

    Supports:
    - Dense decode against compressed KV cache
    - FP8 KV cache via is_fp8_kvcache flag
    - Sparse decode with indices parameter for DSA
    """

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        head_dim_v: int,
        tile_scheduler_metadata: torch.Tensor,
        softmax_scale: float,
        is_fp8_kvcache: bool = False,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return flash_mla_with_kvcache(
            q,
            kv_cache,
            block_table,
            cache_seqlens,
            head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            softmax_scale=softmax_scale,
            is_fp8_kvcache=is_fp8_kvcache,
            indices=indices,
        )


class FlashMLAGetMetadata(nn.Module):
    def forward(self, cache_seqlens: torch.Tensor, num_heads_per_head_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        return get_mla_metadata(cache_seqlens, num_heads_per_head_k)
