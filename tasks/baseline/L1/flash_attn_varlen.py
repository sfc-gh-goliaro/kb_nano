"""Variable-length Flash Attention (no KV cache lookup).

Thin ``nn.Module`` wrapper around ``flash_attn_varlen_func`` with the same
3-way fallback as :mod:`flash_attn_prefill`: vLLM's bundled FA3 (Hopper) / FA4
(Blackwell), then upstream ``flash_attn`` (FA2), then ``flash_mla``.

Used by MLA prefill and chunked-context paths where Q, K, V are dense
``[total_tokens, num_heads, head_dim]`` tensors (no paged cache lookup,
no ``block_table``).  Supports ``return_softmax_lse`` for MLA chunked
prefix merging.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_FA3_AVAILABLE = False
_fa3_varlen_func = None
_fa_version = None
# Dense varlen only — callers of this op (e.g. the embedding engine's
# ``_forward_varlen``) run inside torch.compile, so stay on a dynamo-traceable
# backend.  See ``_fa_backend.TRACEABLE_FA_AVAILABLE``.
from ._fa_backend import FA_VERSION as _fa_backend_version
from ._fa_backend import TRACEABLE_FA_AVAILABLE as _fa_backend_available
from ._fa_backend import fa_version_for_head_dim as _fa_version_for_head_dim
from ._fa_backend import vllm_fa_varlen_func as _fa_backend_varlen

if _fa_backend_available:
    _FA3_AVAILABLE = True
    _fa3_varlen_func = _fa_backend_varlen
    _fa_version = _fa_backend_version

_fa2_varlen_func = None
_flashmla_varlen_func = None
if not _FA3_AVAILABLE:
    try:
        from flash_attn import flash_attn_varlen_func as _fa2_varlen_func
    except ImportError:
        # vLLM vendors FlashMLA; fall back to the standalone ``flash_mla``
        # package when the vendored copy is unavailable.
        try:
            from vllm.third_party.flashmla.flash_mla_interface import (
                flash_attn_varlen_func as _flashmla_varlen_func,
            )
        except ImportError:  # pragma: no cover
            from flash_mla import (  # type: ignore[no-redef]
                flash_attn_varlen_func as _flashmla_varlen_func,
            )


def _handles_diff_headdims(version: int | None) -> bool:
    """Does this FA version take v with a smaller head dim than q natively?

    Only FA3 on Hopper and FA4 do. vLLM applies the same rule
    (``MLACommonImpl._pad_v``); everything else -- including vLLM's own FA2 and
    upstream flash_attn -- rejects it outright with
    "v must have shape (total_k, num_heads_k, head_size)".
    """
    if version == 4:
        return True
    if version == 3 and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_capability()[0] == 9
        except Exception:
            return False
    return False


def _maybe_pad_v(q: torch.Tensor, v: torch.Tensor, version: int | None):
    """Zero-pad v out to q's head dim when the kernel cannot do it itself.

    MLA prefill runs qk at 576 and v at 512. Callers slice the output back to
    ``v_head_dim``, matching vLLM's ``_flash_attn_varlen_diff_headdims``.
    """
    if v.shape[-1] == q.shape[-1] or _handles_diff_headdims(version):
        return v
    return torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]], value=0)


def _unpad_out(out, v_head_dim: int):
    """Trim a padded output back to the real v head dim.

    ``_maybe_pad_v`` widens v so kernels that cannot take a smaller v head dim
    will run; the attention output inherits that width. Callers must not have to
    know whether padding happened -- one MLA path reshapes the result directly
    and broke with "shape '[2, 2048]' is invalid for input of size 6144" (2 x 16
    heads x 192 padded, against 16 x 128 expected). Slice here instead, which is
    a no-op when no padding was applied.
    """
    if isinstance(out, tuple):
        head, *rest = out
        return (_unpad_out(head, v_head_dim), *rest)
    if out.shape[-1] != v_head_dim:
        return out[..., :v_head_dim]
    return out


class FlashAttnVarlen(nn.Module):
    """Variable-length Flash Attention without paged KV cache lookup."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool = True,
        return_softmax_lse: bool = False,
    ):
        if _FA3_AVAILABLE:
            kwargs = dict(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                return_softmax_lse=return_softmax_lse,
            )
            # FA4's Blackwell kernels are head-size limited (MLA prefill runs at
            # head_dim 192/576), so resolve the version from the actual tensor.
            version = _fa_version_for_head_dim(q.shape[-1])
            if version is not None:
                kwargs["fa_version"] = version
            return _unpad_out(
                _fa3_varlen_func(q, k, _maybe_pad_v(q, v, version), **kwargs),
                v.shape[-1],
            )
        fn = _fa2_varlen_func if _fa2_varlen_func is not None else _flashmla_varlen_func
        kwargs = dict(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        v_head_dim = v.shape[-1]
        v = _maybe_pad_v(q, v, None)
        if return_softmax_lse:
            if fn is _fa2_varlen_func:
                # Upstream flash_attn spells this ``return_attn_probs`` and
                # returns (out, softmax_lse, S_dmask); vLLM notes the same
                # difference for its ROCm path.
                out, lse, *_ = fn(q, k, v, return_attn_probs=True, **kwargs)
                return _unpad_out(out, v_head_dim), lse
            kwargs["return_softmax_lse"] = return_softmax_lse
        return _unpad_out(fn(q, k, v, **kwargs), v_head_dim)
