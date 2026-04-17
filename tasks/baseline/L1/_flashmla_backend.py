"""Resolve FlashMLA symbols from vLLM's vendored copy, falling back to
the standalone ``flash_mla`` package if installed.

vLLM ships a fully vendored copy of FlashMLA under
``vllm.third_party.flashmla.flash_mla_interface`` together with its
compiled ``vllm._flashmla_C`` / ``vllm._flashmla_extension_C`` kernels,
so kb_nano does not need to build FlashMLA separately.  The standalone
``flash_mla`` package remains supported as a fallback for environments
that predate the vendored copy.
"""

from __future__ import annotations

try:
    from vllm.third_party.flashmla.flash_mla_interface import (
        flash_mla_with_kvcache,
        get_mla_metadata,
        flash_mla_sparse_fwd,
        flash_attn_varlen_func,
    )
except ImportError:  # pragma: no cover
    from flash_mla import (  # type: ignore[no-redef]
        flash_mla_with_kvcache,
        get_mla_metadata,
        flash_mla_sparse_fwd,
        flash_attn_varlen_func,
    )

# FP8-specific dense decode entry points (vLLM-only). Falls back to ``None``
# when the vendored kernels are not available, which forces the dense FP8
# path to use the generic ``flash_mla_with_kvcache(..., is_fp8_kvcache=True)``
# fallback (matches behaviour on hardware without the FP8-specialized kernel).
try:
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_with_kvcache_fp8,
        get_mla_metadata_dense_fp8,
    )
except ImportError:  # pragma: no cover
    flash_mla_with_kvcache_fp8 = None  # type: ignore[assignment]
    get_mla_metadata_dense_fp8 = None  # type: ignore[assignment]

__all__ = [
    "flash_mla_with_kvcache",
    "get_mla_metadata",
    "flash_mla_sparse_fwd",
    "flash_attn_varlen_func",
    "flash_mla_with_kvcache_fp8",
    "get_mla_metadata_dense_fp8",
]
