"""Semantic PyTorch reference for MXFP4 MoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Mxfp4MoEQuantConfig:
    w1_precision: Any
    w2_precision: Any
    w1_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None


_FP4_E2M1_LUT = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)


def _dequant_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    lut = _FP4_E2M1_LUT.to(blocks.device)

    # ``blocks`` arrives either 4-D as stored in the checkpoint
    # ([E, rows, num_blocks, 16]) or 3-D as the module allocates it
    # ([E, rows, H//2] -- one byte per FP4 pair, blocks not split out).
    # Reshaping straight to 32 only works for the 4-D form; on the 3-D form it
    # raises "shape '[128, 5760, 32]' is invalid".  Fold the packed axis into
    # blocks of 16 bytes first so both layouts take the same path.
    if blocks.ndim == 3:
        e, rows, packed = blocks.shape
        blocks = blocks.reshape(e, rows, packed // 16, 16)

    low = (blocks & 0x0F).long()
    high = ((blocks >> 4) & 0x0F).long()
    unpacked = torch.stack([low, high], dim=-1).reshape(*blocks.shape[:-1], 32)
    values = lut[unpacked]

    scale_float = torch.pow(2.0, scales.float() - 127.0)
    while scale_float.ndim < values.ndim - 1:
        scale_float = scale_float.unsqueeze(-1)
    values = values * scale_float.unsqueeze(-1)
    return values.reshape(*values.shape[:-2], -1).to(dtype)


class _LazyMxfp4Weight:
    """Packed MXFP4 expert weights, dequantized per-expert on demand."""

    def __init__(self, blocks: torch.Tensor, scales: torch.Tensor):
        self.blocks = blocks
        self.scales = scales
        self._cache: dict[int, torch.Tensor] = {}

    @property
    def shape(self):
        e, rows = self.blocks.shape[0], self.blocks.shape[1]
        cols = self.blocks.shape[-1] * 2
        if self.blocks.ndim == 4:
            cols = self.blocks.shape[2] * self.blocks.shape[3] * 2
        return (e, rows, cols)

    # Cap the cache: each gpt-oss expert is ~33 MB in bf16 and a decode step
    # touches only a handful, but an unbounded cache grows to the full 128
    # experts across scenarios and pushes the baseline/candidate pair past
    # HBM.  Evicting keeps peak memory flat without re-dequantizing the
    # experts a single forward actually reuses.
    _CACHE_LIMIT = 8

    def expert(self, idx: int) -> torch.Tensor:
        cached = self._cache.get(idx)
        if cached is None:
            cached = _dequant_mxfp4(
                self.blocks[idx: idx + 1], self.scales[idx: idx + 1],
                dtype=torch.bfloat16,
            )[0]
            if len(self._cache) >= self._CACHE_LIMIT:
                self._cache.pop(next(iter(self._cache)))
            self._cache[idx] = cached
        return cached

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.expert(idx)
        return _dequant_mxfp4(self.blocks[idx], self.scales[idx],
                              dtype=torch.bfloat16)

    def to(self, *a, **k):
        return self


class Mxfp4MoE(nn.Module):
    """MXFP4-quantized MoE using dense PyTorch matmuls after dequantization."""

    @staticmethod
    def prepare_weight(
        quant_tensor: torch.Tensor,
        scale: torch.Tensor,
        num_warps: int = 8,
    ):
        del num_warps
        # Do NOT dequantize all experts here.  gpt-oss has 128 experts and a
        # full bf16 expansion is ~4 TB -- the production path never does this
        # (see the baseline's "No dequantization is performed": it feeds the
        # packed tensors straight to a Triton MXFP4 kernel).  Keep the packed
        # pair and let the forward dequantize only the experts a token routes
        # to, which is what makes the reference runnable at all.
        return _LazyMxfp4Weight(quant_tensor, scale), None

    @staticmethod
    def make_quant_config(
        w1_precision: Any,
        w2_precision: Any,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> Mxfp4MoEQuantConfig:
        return Mxfp4MoEQuantConfig(
            w1_precision=w1_precision,
            w2_precision=w2_precision,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1,
        w2,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        quant_config: Mxfp4MoEQuantConfig,
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        scores = torch.softmax(gating_output.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)

        # Resolve per expert, never all at once: a full bf16 expansion of
        # gpt-oss's 128 experts is terabytes.  ``_LazyMxfp4Weight`` dequantizes
        # (and caches) just the experts the router actually selects.
        def _w(src, idx: int) -> torch.Tensor:
            if hasattr(src, "expert"):
                return src.expert(idx).float()
            return src[idx].float()

        output = torch.zeros_like(hidden_states, dtype=torch.float32)
        x_all = hidden_states.float()

        for token in range(hidden_states.shape[0]):
            for slot in range(topk):
                expert = int(topk_ids[token, slot].item())
                weight = topk_weights[token, slot].float()
                x = x_all[token]
                if apply_router_weight_on_input:
                    x = x * weight

                bias1 = None
                if quant_config.w1_bias is not None:
                    bias1 = quant_config.w1_bias[expert].float()
                gate_up = F.linear(x, _w(w1, expert), bias1)
                gate = gate_up[0::2]
                up = gate_up[1::2]
                gate = gate.clamp(max=7.0)
                up = up.clamp(min=-7.0, max=7.0)
                hidden = (up + 1.0) * gate * torch.sigmoid(1.702 * gate)

                bias2 = None
                if quant_config.w2_bias is not None:
                    bias2 = quant_config.w2_bias[expert].float()
                y = F.linear(hidden, _w(w2, expert), bias2)
                if not apply_router_weight_on_input:
                    y = y * weight
                output[token] += y

        return output.to(hidden_states.dtype)


__all__ = ["Mxfp4MoE", "Mxfp4MoEQuantConfig"]
