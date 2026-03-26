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


@triton.jit
def _store_fp8_mla_rope_fused_kernel(
    kv_c_ptr,           # [num_tokens, kv_lora_rank] bf16
    kv_c_stride,
    k_pe_ptr,           # [num_tokens, qk_rope_head_dim] bf16 (pre-RoPE)
    k_pe_stride,
    cos_sin_cache_ptr,  # [max_pos, rope_dim] (interleaved cos, sin)
    positions_ptr,      # [num_tokens] int64
    cache_ptr,          # [num_blocks, block_size, 656] uint8
    slot_mapping_ptr,   # [num_tokens] int64
    BYTES_PER_TOKEN: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    ROPE_HALF: tl.constexpr,
):
    """Fused RoPE + FP8 quantization + cache write for MLA KV."""
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
        pos = tl.load(positions_ptr + idx)
        pair_offs = tl.arange(0, TILE_SIZE)
        pair_mask = pair_offs < ROPE_HALF

        k_raw_ptr = k_pe_ptr + idx * k_pe_stride
        x_even = tl.load(k_raw_ptr + pair_offs * 2, mask=pair_mask).to(tl.float32)
        x_odd = tl.load(k_raw_ptr + pair_offs * 2 + 1, mask=pair_mask).to(tl.float32)

        cos_ptr = cos_sin_cache_ptr + pos * QK_ROPE_HEAD_DIM
        cos_val = tl.load(cos_ptr + pair_offs, mask=pair_mask).to(tl.float32)
        sin_val = tl.load(cos_ptr + ROPE_HALF + pair_offs, mask=pair_mask).to(tl.float32)

        out_even = x_even * cos_val - x_odd * sin_val
        out_odd = x_odd * cos_val + x_even * sin_val

        rope_offset = base + KV_LORA_RANK + 4 * 4
        rope_dst_ptr = (cache_ptr + rope_offset).to(tl.pointer_type(tl.bfloat16))
        tl.store(rope_dst_ptr + pair_offs * 2, out_even.to(tl.bfloat16), mask=pair_mask)
        tl.store(rope_dst_ptr + pair_offs * 2 + 1, out_odd.to(tl.bfloat16), mask=pair_mask)


class StoreKVCacheFP8MLA(nn.Module):
    """Store MLA latent KV into FP8 packed paged cache (656 bytes/token)."""

    def forward(self, kv_c_normed, k_pe, kv_cache, slot_mapping,
                positions=None, cos_sin_cache=None):
        """
        Args:
            kv_c_normed: [num_tokens, 512] bfloat16 (normed latent)
            k_pe: [num_tokens, 64] bfloat16 (RoPE key, pre- or post-RoPE)
            kv_cache: [num_blocks, block_size, 656] uint8
            slot_mapping: [num_tokens] int64
            positions: [num_tokens] int64 (for fused RoPE, optional)
            cos_sin_cache: [max_pos, 64] bf16 (interleaved cos/sin, optional)
        """
        N = kv_c_normed.shape[0]
        if positions is not None and cos_sin_cache is not None:
            grid = (N, 5)
            _store_fp8_mla_rope_fused_kernel[grid](
                kv_c_normed, kv_c_normed.stride(0),
                k_pe, k_pe.stride(0),
                cos_sin_cache,
                positions,
                kv_cache.view(-1),
                slot_mapping,
                BYTES_PER_TOKEN=BYTES_PER_TOKEN,
                KV_LORA_RANK=KV_LORA_RANK,
                TILE_SIZE=TILE_SIZE,
                QK_ROPE_HEAD_DIM=QK_ROPE_HEAD_DIM,
                ROPE_HALF=QK_ROPE_HEAD_DIM // 2,
            )
        else:
            grid = (N, 5)
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


def gather_fp8_mla_to_bf16(kv_cache, block_table, seq_lens, cum_seq_lens,
                           total_tokens, out_buf, block_size=64):
    """Gather FP8 MLA KV cache and dequantize to BF16.

    Args:
        kv_cache: [num_blocks, block_size, 656] uint8
        block_table: [num_seqs, max_blocks_per_seq] int32
        seq_lens: [num_seqs] int32
        cum_seq_lens: [num_seqs+1] int32
        total_tokens: int
        out_buf: [total_tokens, 576] bf16 (output)
    """
    num_seqs = seq_lens.shape[0]
    max_seq_len = int(seq_lens.max().item()) if num_seqs > 0 else 0
    if max_seq_len == 0:
        return
    grid = (max_seq_len, 5, num_seqs)
    _gather_fp8_to_bf16_kernel[grid](
        kv_cache.view(-1), out_buf, out_buf.stride(0),
        block_table, block_table.stride(0),
        seq_lens, cum_seq_lens,
        BLOCK_SIZE=block_size,
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
