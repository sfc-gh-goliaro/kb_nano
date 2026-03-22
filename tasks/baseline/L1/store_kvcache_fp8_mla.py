"""Triton kernel for storing MLA latent KV into FP8 packed paged cache.

FP8 MLA KV cache format (656 bytes per token):
  - First 512 bytes: 512 float8_e4m3 values (quantized NoPE / kv_c_normed)
  - Next 16 bytes: 4 float32 scale factors (one per 128 FP8 values)
  - Last 128 bytes: 64 bfloat16 values (RoPE part, k_pe, not quantized)

Cache shape: [num_blocks, block_size, 656] dtype=uint8
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

BYTES_PER_TOKEN = 656
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
TILE_SIZE = 128
NUM_TILES = 4


@triton.jit
def _store_fp8_mla_kernel(
    kv_c_ptr,           # [num_tokens, kv_lora_rank] bf16
    kv_c_stride,
    k_pe_ptr,           # [num_tokens, qk_rope_head_dim] bf16
    k_pe_stride,
    cache_ptr,          # [num_blocks, block_size, 656] uint8
    slot_mapping_ptr,   # [num_tokens] int64
    BYTES_PER_TOKEN: tl.constexpr,  # 656
    KV_LORA_RANK: tl.constexpr,    # 512
    TILE_SIZE: tl.constexpr,        # 128
    QK_ROPE_HEAD_DIM: tl.constexpr, # 64
):
    idx = tl.program_id(0)
    tile_id = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return

    base = slot * BYTES_PER_TOKEN

    if tile_id < 4:
        nope_offs = tl.arange(0, TILE_SIZE)
        nope_src = idx * kv_c_stride + tile_id * TILE_SIZE + nope_offs
        vals_bf16 = tl.load(kv_c_ptr + nope_src).to(tl.float32)

        amax = tl.max(tl.abs(vals_bf16))
        scale_inv = amax / 448.0
        scale_inv = tl.where(scale_inv < 1e-12, 1e-12, scale_inv)
        log2_s = tl.math.log2(scale_inv)
        log2_s_ceil = tl.math.ceil(log2_s)
        scale_inv_ue8m0 = tl.math.exp2(log2_s_ceil)

        vals_fp8 = (vals_bf16 / scale_inv_ue8m0).to(tl.float8e4nv)
        vals_u8 = vals_fp8.to(tl.uint8, bitcast=True)

        nope_dst = base + tile_id * TILE_SIZE + nope_offs
        tl.store(cache_ptr + nope_dst, vals_u8)

        scale_offset = base + KV_LORA_RANK + tile_id * 4
        scale_byte_ptr = (cache_ptr + scale_offset).to(tl.pointer_type(tl.float32))
        tl.store(scale_byte_ptr, scale_inv_ue8m0)
    else:
        rope_offs = tl.arange(0, TILE_SIZE)
        rope_mask = rope_offs < QK_ROPE_HEAD_DIM
        rope_src = idx * k_pe_stride + rope_offs
        rope_vals = tl.load(k_pe_ptr + rope_src, mask=rope_mask)

        rope_offset = base + KV_LORA_RANK + 4 * 4
        rope_dst_ptr = (cache_ptr + rope_offset).to(tl.pointer_type(tl.bfloat16))
        tl.store(rope_dst_ptr + rope_offs, rope_vals, mask=rope_mask)


class StoreKVCacheFP8MLA(nn.Module):
    """Store MLA latent KV into FP8 packed paged cache (656 bytes/token)."""

    def forward(self, kv_c_normed, k_pe, kv_cache, slot_mapping):
        """
        Args:
            kv_c_normed: [num_tokens, 512] bfloat16 (normed latent)
            k_pe: [num_tokens, 64] bfloat16 (RoPE key)
            kv_cache: [num_blocks, block_size, 656] uint8
            slot_mapping: [num_tokens] int64
        """
        N = kv_c_normed.shape[0]
        grid = (N, 5)  # 4 NoPE tiles + 1 RoPE tile
        _store_fp8_mla_kernel[grid](
            kv_c_normed, kv_c_normed.stride(0),
            k_pe, k_pe.stride(0),
            kv_cache.view(-1),
            slot_mapping,
            BYTES_PER_TOKEN=BYTES_PER_TOKEN,
            KV_LORA_RANK=KV_LORA_RANK,
            TILE_SIZE=TILE_SIZE,
            QK_ROPE_HEAD_DIM=QK_ROPE_HEAD_DIM,
        )


@triton.jit
def _gather_fp8_to_bf16_kernel(
    cache_ptr,          # [num_blocks * block_size * 656] uint8 flat
    out_ptr,            # [total_tokens, 576] bf16
    out_stride,
    block_table_ptr,    # [num_seqs, max_blocks_per_seq] int32
    bt_stride,
    seq_lens_ptr,       # [num_seqs] int32
    cum_seq_lens_ptr,   # [num_seqs + 1] int32
    BLOCK_SIZE: tl.constexpr,       # 64
    BYTES_PER_TOKEN: tl.constexpr,  # 656
    KV_LORA_RANK: tl.constexpr,     # 512
    TILE_SIZE: tl.constexpr,        # 128
    QK_ROPE_HEAD_DIM: tl.constexpr, # 64
):
    """Gather FP8 KV cache entries, dequantize to BF16 [total_tokens, 576]."""
    tok_idx = tl.program_id(0)
    tile_id = tl.program_id(1)

    num_seqs = tl.num_programs(2)
    seq_id = tl.program_id(2)

    seq_start = tl.load(cum_seq_lens_ptr + seq_id)
    seq_end = tl.load(cum_seq_lens_ptr + seq_id + 1)
    seq_len = seq_end - seq_start

    if tok_idx >= seq_len:
        return

    page_idx = tok_idx // BLOCK_SIZE
    slot_in_page = tok_idx % BLOCK_SIZE
    physical_block = tl.load(block_table_ptr + seq_id * bt_stride + page_idx)
    slot = physical_block * BLOCK_SIZE + slot_in_page
    base = slot * BYTES_PER_TOKEN
    out_row = seq_start + tok_idx

    if tile_id < 4:
        nope_offs = tl.arange(0, TILE_SIZE)
        fp8_ptr = (cache_ptr + base + tile_id * TILE_SIZE).to(tl.pointer_type(tl.float8e4nv))
        vals_fp8 = tl.load(fp8_ptr + nope_offs).to(tl.float32)

        scale_ptr = (cache_ptr + base + KV_LORA_RANK + tile_id * 4).to(tl.pointer_type(tl.float32))
        scale = tl.load(scale_ptr)

        vals_bf16 = (vals_fp8 * scale).to(tl.bfloat16)
        nope_dst = out_row * out_stride + tile_id * TILE_SIZE + nope_offs
        tl.store(out_ptr + nope_dst, vals_bf16)
    else:
        rope_offs = tl.arange(0, TILE_SIZE)
        rope_mask = rope_offs < QK_ROPE_HEAD_DIM
        rope_ptr = (cache_ptr + base + KV_LORA_RANK + 4 * 4).to(tl.pointer_type(tl.bfloat16))
        rope_vals = tl.load(rope_ptr + rope_offs, mask=rope_mask)
        rope_dst = out_row * out_stride + KV_LORA_RANK + rope_offs
        tl.store(out_ptr + rope_dst, rope_vals, mask=rope_mask)
