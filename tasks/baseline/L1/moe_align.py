"""MoE token-to-expert alignment with block padding.

Uses sgl_kernel.moe_align_block_size for high-performance, CUDA-graph-compatible
token-to-expert alignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton

from sgl_kernel import moe_align_block_size as _sgl_moe_align


class MoeAlign(nn.Module):
    """MoE token-to-expert alignment using sgl_kernel.

    Pre-allocates output buffers for reuse and CUDA graph compatibility.
    """

    def __init__(self):
        super().__init__()
        self._sorted_token_ids = None
        self._expert_ids = None
        self._num_tokens_post_padded = None
        self._cumsum_buffer = None

    def _ensure_buffers(self, max_padded, max_blocks, num_experts, device):
        if (self._sorted_token_ids is None
                or self._sorted_token_ids.size(0) < max_padded):
            self._sorted_token_ids = torch.empty(
                max_padded, dtype=torch.int32, device=device,
            )
        if (self._expert_ids is None
                or self._expert_ids.size(0) < max_blocks):
            self._expert_ids = torch.empty(
                max_blocks, dtype=torch.int32, device=device,
            )
        if (self._num_tokens_post_padded is None
                or self._num_tokens_post_padded.device != device):
            self._num_tokens_post_padded = torch.zeros(
                1, dtype=torch.int32, device=device,
            )
        if (self._cumsum_buffer is None
                or self._cumsum_buffer.size(0) < num_experts + 2):
            self._cumsum_buffer = torch.zeros(
                num_experts + 2, dtype=torch.int32, device=device,
            )

    def forward(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        numel = topk_ids.numel()
        if numel < num_experts + 1:
            max_padded = numel * block_size
        else:
            max_padded = numel + (num_experts + 1) * (block_size - 1)
        max_blocks = triton.cdiv(max_padded, block_size)

        self._ensure_buffers(max_padded, max_blocks, num_experts, topk_ids.device)

        sorted_token_ids = self._sorted_token_ids[:max_padded]
        expert_ids = self._expert_ids[:max_blocks]

        _sgl_moe_align(
            topk_ids.view(-1).contiguous(),
            num_experts + 1, block_size,
            sorted_token_ids, expert_ids,
            self._num_tokens_post_padded,
            self._cumsum_buffer[:num_experts + 2],
            True,
        )

        return sorted_token_ids, expert_ids, self._num_tokens_post_padded
