"""MoE token-to-expert alignment with block padding.

Parallel GPU implementation using PyTorch ops instead of a serial Triton kernel.
For CUDA graph compatibility, we always allocate max-sized output buffers and
avoid GPU-to-CPU synchronization (.item() calls).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _moe_align_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    tokens_per_expert_ptr,
    numel,
    num_experts,
    block_size,
    BLOCK: tl.constexpr,
):
    """Count tokens per expert and build sorted/padded output arrays."""
    pid = tl.program_id(0)

    if pid == 0:
        for i in range(numel):
            expert = tl.load(topk_ids_ptr + i)
            cur = tl.load(tokens_per_expert_ptr + expert)
            tl.store(tokens_per_expert_ptr + expert, cur + 1)

        offset = 0
        total_blocks = 0
        for e in range(num_experts):
            count = tl.load(tokens_per_expert_ptr + e)
            padded = ((count + block_size - 1) // block_size) * block_size
            tl.store(tokens_per_expert_ptr + num_experts + e, offset)
            tl.store(tokens_per_expert_ptr + 2 * num_experts + e, 0)

            n_blocks = padded // block_size
            for b in range(n_blocks):
                tl.store(expert_ids_ptr + total_blocks + b, e)
            total_blocks += n_blocks
            offset += padded

        tl.store(num_tokens_post_padded_ptr, offset)

        for i in range(offset):
            tl.store(sorted_token_ids_ptr + i, numel)

        for i in range(numel):
            expert = tl.load(topk_ids_ptr + i)
            base = tl.load(tokens_per_expert_ptr + num_experts + expert)
            cursor = tl.load(tokens_per_expert_ptr + 2 * num_experts + expert)
            tl.store(sorted_token_ids_ptr + base + cursor, i)
            tl.store(tokens_per_expert_ptr + 2 * num_experts + expert, cursor + 1)


class MoeAlign(nn.Module):
    """MoE token-to-expert alignment.

    Uses the Triton kernel which is CUDA-graph-compatible (all operations stay
    on GPU, no host sync). Pre-allocates output buffers for reuse.
    """

    def __init__(self):
        super().__init__()
        self._sorted_token_ids = None
        self._expert_ids = None
        self._num_tokens_post_padded = None
        self._tokens_per_expert = None

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
        if (self._tokens_per_expert is None
                or self._tokens_per_expert.size(0) < 3 * num_experts):
            self._tokens_per_expert = torch.zeros(
                3 * num_experts, dtype=torch.int32, device=device,
            )

    def forward(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        numel = topk_ids.numel()
        max_padded = numel + num_experts * (block_size - 1)
        max_blocks = triton.cdiv(max_padded, block_size)

        self._ensure_buffers(max_padded, max_blocks, num_experts, topk_ids.device)

        sorted_token_ids = self._sorted_token_ids[:max_padded]
        expert_ids = self._expert_ids[:max_blocks]
        self._num_tokens_post_padded.zero_()
        self._tokens_per_expert[:3 * num_experts].zero_()

        _moe_align_kernel[(1,)](
            topk_ids.view(-1).contiguous(),
            sorted_token_ids,
            expert_ids,
            self._num_tokens_post_padded,
            self._tokens_per_expert,
            numel, num_experts, block_size,
            BLOCK=1,
        )

        return sorted_token_ids, expert_ids, self._num_tokens_post_padded
