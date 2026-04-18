"""Tree-mask attention for the EAGLE-3 verify step.

Each sequence has a small window of ``num_verify_tokens`` query tokens that
must attend to the entire prefix (causal) plus a tree-structured mask over the
sibling/parent draft tokens. We implement this with PyTorch SDPA, gathering
the full KV from the paged cache per sequence. Sizes are tiny (B * 64 ≈ a few
thousand tokens), so the per-sequence Python loop is fine.

This op is only used for the EAGLE-3 verify forward; the regular causal prefill
and decode paths are unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_paged_kv(
    cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """Gather first `seq_len` tokens from paged KV cache for one sequence.

    cache: [num_blocks, block_size, H_kv, D] (NHD layout)
    block_table_row: [max_blocks] int32
    Returns: [seq_len, H_kv, D]
    """
    n_blocks = (seq_len + block_size - 1) // block_size
    if n_blocks == 0:
        return cache.new_empty((0, cache.shape[2], cache.shape[3]))
    blocks = block_table_row[:n_blocks].long()
    gathered = cache[blocks]
    flat = gathered.reshape(n_blocks * block_size, cache.shape[2], cache.shape[3])
    return flat[:seq_len]


class TreeAttnPrefill(nn.Module):
    """Verify-step attention with a tree mask, looping over sequences."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        self.gqa_groups = num_heads // num_kv_heads

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        tree_mask: torch.Tensor,
        num_verify_tokens: int,
        block_size: int,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        q : [B * N, H_q, D] -- the new query tokens (N = num_verify_tokens)
        k_cache, v_cache : paged NHD cache, KV for prefix + new draft already stored
        cache_seqlens : [B] -- total seq length **including** the just-stored
            draft tokens (i.e. prefix_len + N)
        block_table : [B, max_blocks_per_seq]
        tree_mask : flat bool tensor in FULL_MASK layout produced by
            build_tree_kernel_efficient. For batch i with prefix length
            s_i = cache_seqlens[i] - N it contains N rows of (s_i + N) bools,
            concatenated across the batch.
        """
        scale = softmax_scale if softmax_scale is not None else self.sm_scale
        B = cache_seqlens.shape[0]
        N = num_verify_tokens
        H_q = self.num_heads
        H_kv = self.num_kv_heads
        D = self.head_dim
        device = q.device
        dtype = q.dtype

        out = torch.empty_like(q)

        cs_cpu = cache_seqlens.tolist()
        mask_offset = 0
        for i in range(B):
            total_len = int(cs_cpu[i])
            prefix_len = total_len - N
            k_full = _gather_paged_kv(k_cache, block_table[i], total_len, block_size)
            v_full = _gather_paged_kv(v_cache, block_table[i], total_len, block_size)

            row_len = prefix_len + N
            mask_size = N * row_len
            mask_bits = tree_mask[mask_offset:mask_offset + mask_size].view(N, row_len)
            mask_offset += mask_size

            qi = q[i * N:(i + 1) * N]
            qi_b = qi.transpose(0, 1).unsqueeze(0)
            k_b = k_full.transpose(0, 1).unsqueeze(0)
            v_b = v_full.transpose(0, 1).unsqueeze(0)

            attn_mask = mask_bits.to(dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0)

            oi = F.scaled_dot_product_attention(
                qi_b, k_b, v_b,
                attn_mask=attn_mask,
                dropout_p=0.0,
                scale=scale,
                enable_gqa=(self.gqa_groups != 1),
            )
            out[i * N:(i + 1) * N] = oi.squeeze(0).transpose(0, 1).to(dtype)

        return out
