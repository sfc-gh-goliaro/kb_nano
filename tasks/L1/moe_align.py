"""MoE token-to-expert alignment with block padding (Triton kernel)."""

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
    def forward(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        numel = topk_ids.numel()
        max_padded = numel + num_experts * (block_size - 1)
        max_blocks = triton.cdiv(max_padded, block_size)

        sorted_token_ids = torch.full(
            (max_padded,), numel, dtype=torch.int32, device=topk_ids.device,
        )
        expert_ids = torch.full(
            (max_blocks,), 0, dtype=torch.int32, device=topk_ids.device,
        )
        num_tokens_post_padded = torch.zeros(1, dtype=torch.int32, device=topk_ids.device)
        tokens_per_expert = torch.zeros(
            3 * num_experts, dtype=torch.int32, device=topk_ids.device,
        )

        _moe_align_kernel[(1,)](
            topk_ids.view(-1).contiguous(),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            tokens_per_expert,
            numel, num_experts, block_size,
            BLOCK=1,
        )

        return sorted_token_ids, expert_ids, num_tokens_post_padded
