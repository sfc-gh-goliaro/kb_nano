"""Convert request-local token indices to global paged cache slots."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _convert_req_index_to_global_index_kernel(
    indices_ptr,
    block_table_ptr,
    output_ptr,
    block_size: tl.constexpr,
    num_seqs: tl.constexpr,
    topk: tl.constexpr,
    stride_bt_seq,
    stride_bt_block,
):
    pid = tl.program_id(0)
    seq_idx = pid // topk
    k_idx = pid % topk

    if seq_idx >= num_seqs:
        return

    local_idx = tl.load(indices_ptr + seq_idx * topk + k_idx)

    if local_idx < 0:
        tl.store(output_ptr + seq_idx * topk + k_idx, -1)
        return

    block_idx = local_idx // block_size
    slot_in_block = local_idx % block_size

    physical_block = tl.load(
        block_table_ptr + seq_idx * stride_bt_seq + block_idx * stride_bt_block,
    )

    global_slot = physical_block * block_size + slot_in_block
    tl.store(output_ptr + seq_idx * topk + k_idx, global_slot)


class ConvertIndicesToGlobal(nn.Module):
    """Map per-request token indices to global linear cache slots."""

    def forward(
        self,
        indices: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Convert local indices to global slot indices.

        Args:
            indices: ``[num_seqs, topk]`` int32 — per-request token indices.
            block_table: ``[num_seqs, max_blocks]`` int32 — physical blocks.
            block_size: tokens per block.

        Returns:
            ``global_indices``: ``[num_seqs, topk]`` int32 — linear slots; ``-1``
            when the input index is negative.
        """
        num_seqs, topk = indices.shape
        output = torch.empty_like(indices)

        grid = (num_seqs * topk,)
        _convert_req_index_to_global_index_kernel[grid](
            indices,
            block_table,
            output,
            block_size=block_size,
            num_seqs=num_seqs,
            topk=topk,
            stride_bt_seq=block_table.stride(0),
            stride_bt_block=block_table.stride(1),
        )

        return output
