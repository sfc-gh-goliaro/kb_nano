"""Rotary position embedding for diffusion models (interleaved / GPT-J style).

Pure-PyTorch semantic reference for the baseline's Triton ``_rotary_kernel``
(from ``flash_attn.ops.triton.rotary``, Tri Dao 2023, via
``vllm.vllm_flash_attn.ops.triton.rotary``).

The ``_apply_rotary`` contract is preserved: both head-dim layouts, a
``rotary_dim`` smaller than ``headdim`` (the tail is copied through
untouched), scalar or per-batch ``seqlen_offsets``, the varlen packed layout,
``conjugate`` (negated sine, for the backward direction), and in-place
operation.

Rotation is computed in fp32 and cast back once, which is what the kernel
does via its fp32 accumulators.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn


def _rotate(
    x_rot: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool,
    conjugate: bool,
) -> torch.Tensor:
    """Apply the rotation to the leading ``rotary_dim`` channels.

    Args:
        x_rot: ``[..., rotary_dim]``
        cos, sin: broadcastable to ``[..., rotary_dim // 2]``
    """
    orig_dtype = x_rot.dtype
    xf = x_rot.to(torch.float32)
    cosf = cos.to(torch.float32)
    sinf = sin.to(torch.float32)
    if conjugate:
        sinf = -sinf

    if interleaved:
        # GPT-J: consecutive channel pairs (0,1), (2,3), ...
        x1 = xf[..., 0::2]
        x2 = xf[..., 1::2]
        o1 = x1 * cosf - x2 * sinf
        o2 = x1 * sinf + x2 * cosf
        out = torch.stack((o1, o2), dim=-1).flatten(-2)
    else:
        # GPT-NeoX: split the rotary block in half
        half = xf.shape[-1] // 2
        x1 = xf[..., :half]
        x2 = xf[..., half:]
        o1 = x1 * cosf - x2 * sinf
        o2 = x1 * sinf + x2 * cosf
        out = torch.cat((o1, o2), dim=-1)

    return out.to(orig_dtype)


def _apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """Apply rotary embeddings, mirroring the Triton launcher's contract.

    Args:
        x: (batch, seqlen, nheads, headdim) or (total_seqlen, nheads, headdim)
            if ``cu_seqlens`` is provided.
        cos, sin: (seqlen_ro, rotary_dim / 2)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim
    assert headdim <= 256
    assert seqlen_ro >= seqlen

    cos, sin = cos.contiguous(), sin.contiguous()

    output = x if inplace else torch.empty_like(x)
    if not inplace and rotary_dim < headdim:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    def _offset_for(batch_idx: int) -> int:
        if isinstance(seqlen_offsets, torch.Tensor):
            return int(seqlen_offsets[batch_idx].item())
        return int(seqlen_offsets)

    if not is_varlen:
        for b in range(batch):
            off = _offset_for(b)
            pos = torch.arange(seqlen, device=x.device) + off
            # [seqlen, 1, rotary_dim // 2] -> broadcasts over heads
            c = cos[pos].unsqueeze(1)
            s = sin[pos].unsqueeze(1)
            rotated = _rotate(
                x[b, :, :, :rotary_dim], c, s, interleaved, conjugate,
            )
            output[b, :, :, :rotary_dim] = rotated
    else:
        for b in range(batch):
            start = int(cu_seqlens[b].item())
            end = int(cu_seqlens[b + 1].item())
            if end <= start:
                continue
            off = _offset_for(b)
            pos = torch.arange(end - start, device=x.device) + off
            c = cos[pos].unsqueeze(1)
            s = sin[pos].unsqueeze(1)
            rotated = _rotate(
                x[start:end, :, :rotary_dim], c, s, interleaved, conjugate,
            )
            output[start:end, :, :rotary_dim] = rotated

    return output


class DiffusionRoPE(nn.Module):
    """Apply rotary embeddings given pre-computed (cos, sin) tensors.

    Parameters
    ----------
    is_neox_style : bool
        If True, use the GPT-NeoX (half-split) layout.
        If False (default for FLUX), use the interleaved (GPT-J) layout.
    """

    def __init__(self, is_neox_style: bool = False) -> None:
        super().__init__()
        self.interleaved = not is_neox_style

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if cos.dim() == 3:
            cos = cos[0]
            sin = sin[0]
        return _apply_rotary(x, cos, sin, interleaved=self.interleaved)
