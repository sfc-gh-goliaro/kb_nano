"""Resolve the fastest vLLM-bundled FlashAttention version for the current GPU.

vLLM ships FA3 kernels for Hopper (sm90) and FA4 (CuTe-DSL) kernels for
Blackwell (sm100+).  ``is_fa_version_supported(3)`` is therefore ``False`` on
B200, which used to send the L1 attention ops to the upstream ``flash_attn``
(FA2) fallback.  That fallback is fine for dense varlen, but FA2's
``mha_varlen_fwd`` rejects a paged KV cache whose block size is not a multiple
of 256 ("Paged KV cache block size must be divisible by 256"), which the engine
never uses — so Whisper's paged cross-attention prefill crashed on Blackwell.

This module centralizes the version probe so every L1 op picks FA3 on Hopper and
FA4 on Blackwell, with the FA2 fallback reserved for GPUs that have neither.

``FA_VERSION`` is the value to pass as ``fa_version=`` to
:func:`vllm.vllm_flash_attn.flash_attn_varlen_func`.
"""

from __future__ import annotations

import torch

VLLM_FA_AVAILABLE = False
vllm_fa_varlen_func = None
FA_VERSION: int | None = None

try:
    from vllm.vllm_flash_attn import (
        flash_attn_varlen_func as _vllm_fa_varlen,
        is_fa_version_supported,
    )

    if torch.cuda.is_available():
        _cc = torch.cuda.get_device_capability()
        # Prefer the newest kernel family the running GPU + build actually has.
        # sm90 -> FA3, sm100+ -> FA4.  Probe in descending order so a Blackwell
        # build that also ships FA3 still takes FA4.
        for _candidate in (4, 3):
            try:
                _ok = is_fa_version_supported(_candidate)
            except Exception:
                _ok = False
            if _ok:
                FA_VERSION = _candidate
                break
        if FA_VERSION is not None and _cc[0] >= 9:
            VLLM_FA_AVAILABLE = True
            vllm_fa_varlen_func = _vllm_fa_varlen
except ImportError:  # pragma: no cover - vLLM is optional at runtime
    pass


def fa4_active() -> bool:
    """True when the resolved backend is FA4 (Blackwell CuTe-DSL kernels)."""
    return VLLM_FA_AVAILABLE and FA_VERSION == 4


def fa_version_for_head_dim(head_dim: int | None) -> int | None:
    """Resolve the FA version for a given head size.

    FA4's Blackwell kernels are limited by TMEM capacity to ``head_size <= 128``
    (plus 192, the MLA differing-head-dim case); larger head sizes overflow the
    sm100a shared-memory limit at launch. vLLM applies the same rule in
    ``v1/attention/backends/fa_utils.get_flash_attn_version``, so mirror it here
    rather than letting a head_dim-256 model (Qwen3-Next) die inside the kernel.
    """
    if not VLLM_FA_AVAILABLE:
        return None
    if FA_VERSION == 4 and head_dim is not None and head_dim > 128 and head_dim != 192:
        return 2
    return FA_VERSION


# FA4's entry point is a CuTe-DSL @cute.jit function that reads
# ``torch.cuda.current_stream().cuda_stream``.  Under torch.compile, dynamo
# hands it a generic ``torch.Stream`` proxy with no ``cuda_stream`` attribute,
# so tracing the call raises AttributeError.  Ops that run inside a compiled
# region must therefore stay on a traceable backend: FA3 on Hopper, upstream
# FA2 on Blackwell.  FA2 is only inadequate for *paged* KV (its varlen kernel
# demands a page size divisible by 256), and no compiled region uses paged KV.
TRACEABLE_FA_AVAILABLE = VLLM_FA_AVAILABLE and FA_VERSION == 3
