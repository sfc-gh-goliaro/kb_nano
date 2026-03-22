"""Triton kernel for storing key/value into paged KV cache.

Supports two layouts:
  NHD: [num_blocks, block_size, num_kv_heads, head_dim]  (flash_attn path)
  HND: [num_blocks, num_kv_heads, block_size, head_dim]  (TRTLLM path)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _store_kvcache_kernel(
    key_ptr, key_stride, value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    real_D,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offs = tl.arange(0, D)
    mask = offs < real_D
    key = tl.load(key_ptr + idx * key_stride + offs, mask=mask)
    value = tl.load(value_ptr + idx * value_stride + offs, mask=mask)
    tl.store(k_cache_ptr + slot * real_D + offs, key, mask=mask)
    tl.store(v_cache_ptr + slot * real_D + offs, value, mask=mask)


@triton.jit
def _store_kvcache_hnd_kernel(
    key_ptr, key_stride_n, value_ptr, value_stride_n,
    k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    real_head_dim,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Store KV into HND layout [num_blocks, num_kv_heads, block_size, head_dim]."""
    idx = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    block_idx = slot // PAGE_SIZE
    slot_in_block = slot % PAGE_SIZE
    offs = tl.arange(0, HEAD_DIM)
    mask = offs < real_head_dim
    src_k_offset = idx * key_stride_n + head * real_head_dim + offs
    src_v_offset = idx * value_stride_n + head * real_head_dim + offs
    dst_offset = (block_idx * NUM_KV_HEADS * PAGE_SIZE * real_head_dim
                  + head * PAGE_SIZE * real_head_dim
                  + slot_in_block * real_head_dim
                  + offs)
    k = tl.load(key_ptr + src_k_offset, mask=mask)
    v = tl.load(value_ptr + src_v_offset, mask=mask)
    tl.store(k_cache_ptr + dst_offset, k, mask=mask)
    tl.store(v_cache_ptr + dst_offset, v, mask=mask)


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class StoreKVCache(nn.Module):
    """NHD layout store: [num_blocks, block_size, num_kv_heads, head_dim]."""
    def forward(self, key, value, k_cache, v_cache, slot_mapping):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        D_padded = _next_power_of_2(D)
        _store_kvcache_kernel[(N,)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping, D, D_padded,
        )


class StoreKVCacheHND(nn.Module):
    """HND layout store: [num_blocks, num_kv_heads, block_size, head_dim]."""
    def __init__(self, page_size: int):
        super().__init__()
        self.page_size = page_size

    def forward(self, key, value, k_cache, v_cache, slot_mapping):
        N, num_kv_heads, head_dim = key.shape
        head_dim_padded = _next_power_of_2(head_dim)
        _store_kvcache_hnd_kernel[(N, num_kv_heads)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping, head_dim,
            PAGE_SIZE=self.page_size,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim_padded,
        )
