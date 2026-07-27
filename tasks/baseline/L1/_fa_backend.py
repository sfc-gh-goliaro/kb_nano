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

import os

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


# ---------------------------------------------------------------------------
# Opaque FA wrapper for compiled regions on Blackwell
# ---------------------------------------------------------------------------
# The rule above costs real throughput rather than just kernel choice: on B200
# every compiled region falls back to upstream FA2, whose generic
# ``flash_fwd_kernel`` is far slower than the sm100 kernel vLLM itself uses. In
# an nsys trace of the BGE-M3 long-document encoder (both engines in one trace)
# FA2 averaged 857us per call against 389us for vLLM's
# ``flash_fwd_sm100`` on the same shapes -- a 2.2x gap on the kernel that is 39%
# of total GPU time, which is the whole of that row's 0.640x deficit.
#
# Dynamo does not need to trace *into* the kernel, only to know its shape
# behaviour. Registering the call as a custom op with a fake implementation makes
# it opaque, so FA4 becomes usable inside compiled regions and the FA2 fallback
# is no longer needed there. Hopper is unaffected: FA_VERSION == 3 is already
# traceable and keeps its existing path.
OPAQUE_FA_AVAILABLE = False

# Opt-in, because with repeats it loses on every row measured so far. The nsys
# per-call figures above (857us FA2 vs 389us sm100) did not translate into
# end-to-end throughput; 1-wide medians over three runs each:
#
#   row                     FA2 (off)              FA4 (on)
#   ColBERTv2               2.086 (1.474-2.523)    1.297 (1.256-1.329)
#   BGE-M3                  0.763 (0.626-0.901)    0.549 (0.513-0.586)
#
# On BGE-M3 the two ranges do not overlap, so this is not noise -- the single
# 0.978 reading that first motivated the change was an outlier. The mechanism is
# kept because it is correct and verified (bit-identical under fullgraph=True),
# and a workload with long enough sequences may still benefit, but nothing
# enables it by default until a row is measured to win.
#
# ``FASTKERNELS_FA_OPAQUE=1`` enables it; ``FASTKERNELS_FA_OPAQUE_MIN_SEQ`` then
# bounds it to sequences at least that long.
if (VLLM_FA_AVAILABLE and FA_VERSION == 4
        and os.environ.get("FASTKERNELS_FA_OPAQUE", "0") == "1"):
    _fa_lib = torch.library.Library("fastkernels_fa", "FRAGMENT")
    _fa_lib.define(
        "varlen(Tensor q, Tensor k, Tensor v, Tensor cu_seqlens_q, "
        "Tensor cu_seqlens_k, SymInt max_seqlen_q, SymInt max_seqlen_k, "
        "float softmax_scale, bool causal, int fa_version) -> Tensor"
    )

    def _varlen_cuda(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                     max_seqlen_k, softmax_scale, causal, fa_version):
        return vllm_fa_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_k),
            softmax_scale=softmax_scale,
            causal=causal,
            fa_version=fa_version,
        )

    def _varlen_fake(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                     max_seqlen_k, softmax_scale, causal, fa_version):
        # Output carries q's token/head layout with v's head dim; v is padded to
        # q's head dim by the caller, so this is q's shape in practice, but derive
        # it from v so a differing-head-dim call cannot silently mismatch.
        return q.new_empty((*q.shape[:-1], v.shape[-1]))

    _fa_lib.impl("varlen", _varlen_cuda, "CUDA")
    torch.library.register_fake("fastkernels_fa::varlen", _varlen_fake)
    OPAQUE_FA_AVAILABLE = True


def fa_varlen_opaque(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                     max_seqlen_k, softmax_scale, causal, fa_version):
    """FA4 varlen behind a custom op, safe to call from a compiled region."""
    return torch.ops.fastkernels_fa.varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        softmax_scale, causal, fa_version,
    )


# FA4 only wins once the kernel is long enough to amortize its dispatch cost.
# Its CuTe-DSL entry point does per-call Python work, so on short sequences the
# many-small-calls regime is dominated by that overhead rather than by the
# attention itself. Measured 1-wide on B200 with the two embedding rows, which sit
# on opposite sides of this:
#   BGE-M3    (MLDR docs, max_length 8192)      FA2 0.640x -> FA4 0.978x
#   ColBERTv2 (MS MARCO passages, ~180 tokens)  FA2 3.047x -> FA4 0.779x
# So an unconditional switch trades a 1.5x gain on one row for a 3.9x loss on the
# other. Gate on sequence length instead. The default sits well above ColBERT's
# passages and well below BGE-M3's 8192-token documents: at a 1024 threshold
# ColBERT only recovered to 2.268x (some of its batches still cleared it), so the
# bar is set higher rather than tuned to the edge.
# ``FASTKERNELS_FA_OPAQUE_MIN_SEQ`` overrides it.
FA_OPAQUE_MIN_SEQ = int(os.environ.get("FASTKERNELS_FA_OPAQUE_MIN_SEQ", "4096"))


def use_opaque_fa(max_seqlen: int | None) -> bool:
    """True when the opaque FA4 path should be preferred for this call."""
    if not OPAQUE_FA_AVAILABLE:
        return False
    return max_seqlen is not None and max_seqlen >= FA_OPAQUE_MIN_SEQ
