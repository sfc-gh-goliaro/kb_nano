"""FP8 KV cache store and gather for MLA (656 bytes/token).

Cache format per token (656 bytes):

  * ``[0:512]`` — ``kv_c_normed`` as FP512 (``float8_e4m3fn``).
  * ``[512:528]`` — four per-group FP32 UE8M0 scales (128 dims per group).
  * ``[528:656]`` — ``k_pe`` as 64 ``bfloat16`` values (128 bytes).

Cache tensor shape: ``[num_blocks, block_size, 656]`` with ``dtype=torch.uint8``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_KV_C_DIM = 512
_KV_C_FP8_BYTES = 512
_KV_C_SCALE_BYTES = 16
_K_PE_DIM = 64
_K_PE_BYTES = 128
_BYTES_PER_TOKEN = _KV_C_FP8_BYTES + _KV_C_SCALE_BYTES + _K_PE_BYTES
_GROUP_SIZE = 128
_NUM_GROUPS = _KV_C_DIM // _GROUP_SIZE


def _ue8m0_scale(absmax: torch.Tensor) -> torch.Tensor:
    """Per-group UE8M0 scale: ``2 ** ceil(log2(absmax / fp8_max))``."""
    absmax = absmax.clamp(min=1e-12)
    return torch.exp2(torch.ceil(torch.log2(absmax / _FP8_MAX)))


class StoreKVCacheFP8MLA(nn.Module):
    """Store ``kv_c_normed`` and ``k_pe`` into FP8 MLA paged cache.

    Args:
        kv_c_normed: ``[N, 512]`` BF16 — compressed KV after layernorm.
        k_pe: ``[N, 1, 64]`` or ``[N, 64]`` BF16 — RoPE key component.
        kv_cache: ``[num_blocks, block_size, 656]`` uint8.
        slot_mapping: ``[N]`` int32 — linear slot index per token (``-1`` skips).
    """

    def forward(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        N = kv_c_normed.shape[0]
        k_pe_2d = k_pe.reshape(N, -1)
        if k_pe_2d.shape[1] != _K_PE_DIM:
            raise ValueError(f"k_pe last dim must be {_K_PE_DIM}, got {k_pe_2d.shape[1]}")

        cache_flat = kv_cache.view(-1, _BYTES_PER_TOKEN)

        for i in range(N):
            s = int(slot_mapping[i].item())
            if s < 0:
                continue

            kv_c = kv_c_normed[i]

            for g in range(_NUM_GROUPS):
                group = kv_c[g * _GROUP_SIZE:(g + 1) * _GROUP_SIZE].float()
                absmax = group.abs().max()
                scale = _ue8m0_scale(absmax)
                scaled = (group / scale).clamp(-_FP8_MAX, _FP8_MAX)
                fp8_vals = scaled.to(torch.float8_e4m3fn)

                offset = g * _GROUP_SIZE
                cache_flat[s, offset:offset + _GROUP_SIZE] = fp8_vals.view(torch.uint8)

                scale_offset = _KV_C_FP8_BYTES + g * 4
                scale_b = scale.reshape(1).to(dtype=torch.float32).view(torch.uint8)
                cache_flat[s, scale_offset:scale_offset + 4] = scale_b

            pe_offset = _KV_C_FP8_BYTES + _KV_C_SCALE_BYTES
            pe_bytes = k_pe_2d[i].contiguous().view(torch.uint8)
            cache_flat[s, pe_offset:pe_offset + _K_PE_BYTES] = pe_bytes


class GatherKVCacheFP8MLA(nn.Module):
    """Gather and dequantize KV from FP8 MLA paged cache to BF16.

    Returns:
        ``kv_c_normed_bf16``: ``[total_tokens, 512]`` BF16.
        ``k_pe_bf16``: ``[total_tokens, 64]`` BF16.
    """

    def forward(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = kv_cache.device
        block_size = kv_cache.shape[1]
        total_tokens = int(cu_seq_lens[-1].item())

        cache_flat = kv_cache.view(-1, _BYTES_PER_TOKEN)

        kv_c_out = torch.empty(
            total_tokens, _KV_C_DIM, dtype=torch.bfloat16, device=device,
        )
        k_pe_out = torch.empty(
            total_tokens, _K_PE_DIM, dtype=torch.bfloat16, device=device,
        )

        num_seqs = cu_seq_lens.shape[0] - 1
        out_idx = 0
        for seq_i in range(num_seqs):
            seq_start = int(cu_seq_lens[seq_i].item())
            seq_end = int(cu_seq_lens[seq_i + 1].item())
            seq_len = seq_end - seq_start

            for t in range(seq_len):
                block_idx = t // block_size
                slot_in_block = t % block_size
                physical_block = int(block_table[seq_i, block_idx].item())
                slot = physical_block * block_size + slot_in_block

                for g in range(_NUM_GROUPS):
                    fp8_offset = g * _GROUP_SIZE
                    fp8_bytes = cache_flat[slot, fp8_offset:fp8_offset + _GROUP_SIZE]
                    fp8_vals = fp8_bytes.view(torch.float8_e4m3fn).float()

                    scale_offset = _KV_C_FP8_BYTES + g * 4
                    scale = cache_flat[slot, scale_offset:scale_offset + 4].view(
                        torch.float32,
                    )

                    kv_c_out[out_idx, g * _GROUP_SIZE:(g + 1) * _GROUP_SIZE] = (
                        fp8_vals * scale
                    ).to(torch.bfloat16)

                pe_offset = _KV_C_FP8_BYTES + _KV_C_SCALE_BYTES
                pe_bytes = cache_flat[slot, pe_offset:pe_offset + _K_PE_BYTES]
                k_pe_out[out_idx] = pe_bytes.view(torch.bfloat16)

                out_idx += 1

        return kv_c_out, k_pe_out
