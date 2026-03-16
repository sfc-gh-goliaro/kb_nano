"""Vision encoder rotary position embeddings.

Precomputes a cos/sin cache from fixed inv_freq (base=10000, no scaling).
forward() builds 2D (height, width) position IDs from grid_thw metadata,
shuffled by spatial_merge_size, and returns (cos, sin) tensors ready for
flash_attn's apply_rotary.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, max_grid_size: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
        ))
        t = torch.arange(max_grid_size, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        grid_thw_list: list[list[int]],
        spatial_merge_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sms = spatial_merge_size
        pos_parts = []
        max_grid_size = 0
        for t, h, w in grid_thw_list:
            hpos = np.arange(h, dtype=np.int32).reshape(h, 1).repeat(w, axis=1)
            wpos = np.arange(w, dtype=np.int32).reshape(1, w).repeat(h, axis=0)
            hpos = hpos.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).reshape(-1)
            wpos = wpos.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).reshape(-1)
            hw = np.stack([hpos, wpos], axis=-1)
            if t > 1:
                hw = np.tile(hw, (t, 1))
            pos_parts.append(hw)
            if h > max_grid_size:
                max_grid_size = h
            if w > max_grid_size:
                max_grid_size = w
        pos_ids = torch.from_numpy(np.concatenate(pos_parts, axis=0)).to(device)

        cache = self.cos_sin_cache[:max_grid_size].to(dtype=dtype)
        cos, sin = cache.chunk(2, dim=-1)
        return cos[pos_ids].flatten(1), sin[pos_ids].flatten(1)
