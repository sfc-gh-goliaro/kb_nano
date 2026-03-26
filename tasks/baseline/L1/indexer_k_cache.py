"""Indexer K cache store and gather for DSA (132 bytes/token).

Cache format per token (132 bytes):

  * ``[0:128]`` — K as ``float8_e4m3fn``.
  * ``[128:132]`` — one float32 UE8M0 scale for the full head.

Cache tensor shape: ``[num_blocks, block_size, 132]`` with ``dtype=torch.uint8``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_HEAD_DIM = 128
_SCALE_BYTES = 4
_BYTES_PER_TOKEN = _HEAD_DIM + _SCALE_BYTES
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _ue8m0_scale(absmax: torch.Tensor) -> torch.Tensor:
    absmax = absmax.clamp(min=1e-12)
    return torch.exp2(torch.ceil(torch.log2(absmax / _FP8_MAX)))


class IndexerKCacheStore(nn.Module):
    """Quantize K to FP8 and store in indexer paged cache.

    Args:
        k: ``[N, head_dim]`` BF16 — indexer key vectors.
        kv_cache: ``[num_blocks, block_size, 132]`` uint8.
        slot_mapping: ``[N]`` int32 — linear slot index per token.
    """

    def forward(
        self,
        k: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        N, D = k.shape
        if D != _HEAD_DIM:
            raise ValueError(f"head_dim must be {_HEAD_DIM}, got {D}")

        cache_flat = kv_cache.view(-1, _BYTES_PER_TOKEN)

        for i in range(N):
            s = int(slot_mapping[i].item())
            if s < 0:
                continue

            vals = k[i].float()
            absmax = vals.abs().max()
            scale = _ue8m0_scale(absmax)

            fp8_vals = (vals / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
            cache_flat[s, :_HEAD_DIM] = fp8_vals.view(torch.uint8)

            scale_b = scale.reshape(1).to(dtype=torch.float32).view(torch.uint8)
            cache_flat[s, _HEAD_DIM:_HEAD_DIM + _SCALE_BYTES] = scale_b


class IndexerKCacheGather(nn.Module):
    """Gather K from indexer paged cache in FP8 form.

    Returns:
        ``k_fp8``: ``[total_tokens, head_dim]`` float8_e4m3fn.
        ``k_scale``: ``[total_tokens]`` float32.
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

        k_fp8 = torch.empty(
            total_tokens, _HEAD_DIM, dtype=torch.float8_e4m3fn, device=device,
        )
        k_scale = torch.empty(total_tokens, dtype=torch.float32, device=device)

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

                k_fp8[out_idx] = cache_flat[slot, :_HEAD_DIM].view(torch.float8_e4m3fn)
                k_scale[out_idx] = cache_flat[
                    slot, _HEAD_DIM:_HEAD_DIM + _SCALE_BYTES
                ].view(torch.float32).item()
                out_idx += 1

        return k_fp8, k_scale
