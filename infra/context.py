"""Global inference context for paged KV cache coordination.

The ``Context`` dataclass carries per-step metadata used by attention, MoE,
CUDA graph capture, and torch.compile.  The ``no_compile_layers`` dict
mirrors vLLM's ``ForwardContext.no_compile_layers`` / ``static_forward_context``
so that custom ops can resolve their target module at runtime without baking
references into the compiled graph.
"""

from __future__ import annotations

import enum
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn


class CUDAGraphMode(enum.IntEnum):
    """Runtime mode for CUDA graph dispatch (mirrors vLLM CUDAGraphMode)."""
    NONE = 0
    PIECEWISE = 1
    FULL = 2


@dataclass(frozen=True)
class AttnBackendConfig:
    """Selects attention backend and associated KV cache parameters.

    Blackwell (sm_100+) uses TRTLLM-gen kernels via FlashInfer (HND layout,
    block_size=16).  Hopper and below use flash_attn (NHD layout,
    block_size=256).  Auto-detection picks the optimal backend for the
    current GPU.
    """
    backend: str = "flash_attn"
    block_size: int = 256
    kv_layout: str = "NHD"

    @classmethod
    def auto_detect(cls) -> "AttnBackendConfig":
        if not torch.cuda.is_available():
            return cls()
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 10:
            try:
                from flashinfer.decode import trtllm_batch_decode_with_kv_cache  # noqa: F401
                return cls(backend="trtllm", block_size=16, kv_layout="HND")
            except ImportError:
                pass
        return cls()

    @property
    def use_trtllm(self) -> bool:
        return self.backend == "trtllm"


_ATTN_BACKEND_CONFIG: AttnBackendConfig | None = None


def get_attn_backend_config() -> AttnBackendConfig:
    global _ATTN_BACKEND_CONFIG
    if _ATTN_BACKEND_CONFIG is None:
        _ATTN_BACKEND_CONFIG = AttnBackendConfig.auto_detect()
    return _ATTN_BACKEND_CONFIG


def set_attn_backend_config(config: AttnBackendConfig) -> None:
    global _ATTN_BACKEND_CONFIG
    _ATTN_BACKEND_CONFIG = config


@dataclass
class ChunkedContextMetadata:
    """Metadata for chunked prefill context processing (MLA).

    When a prefill request has prior computed tokens in the KV cache,
    those tokens must be gathered and attended to in chunks to bound
    workspace memory.  Matches vllm's MLACommonPrefillMetadata.ChunkedContextMetadata.
    """
    cu_seq_lens: torch.Tensor
    starts: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]
    seq_lens: torch.Tensor
    workspace: torch.Tensor
    token_to_seq: torch.Tensor
    chunk_total_token: list[int]


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    max_context_len: int = 0

    # Chunked prefill: mixed batch with both prefill and decode tokens.
    # Token layout: [prefill_tokens... | decode_tokens...]
    is_mixed: bool = False
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefill_seqs: int = 0

    # Prefill-specific metadata (indexed over prefill seqs only)
    prefill_cu_seqlens_q: torch.Tensor | None = None
    prefill_cu_seqlens_k: torch.Tensor | None = None
    prefill_max_seqlen_q: int = 0
    prefill_max_seqlen_k: int = 0
    prefill_block_tables: torch.Tensor | None = None

    # Decode-specific metadata (indexed over decode seqs only)
    decode_context_lens: torch.Tensor | None = None
    decode_block_tables: torch.Tensor | None = None
    decode_max_context_len: int = 0

    # Flat indices into concatenated input for extracting one logit per seq
    logit_indices: torch.Tensor | None = None

    # MLA chunked prefill context (for requests with prior computed tokens)
    chunked_context: ChunkedContextMetadata | None = None

    # Per-token request ID mapping (for sparse indexer index conversion)
    req_id_per_token: torch.Tensor | None = None

    # Cross-attention metadata (encoder-decoder models like Whisper)
    # Slot mapping for writing encoder K/V to paged cache
    cross_slot_mapping: torch.Tensor | None = None
    # Prefill: cu_seqlens for decoder Q and encoder K
    cross_cu_seqlens_q: torch.Tensor | None = None
    cross_cu_seqlens_k: torch.Tensor | None = None
    cross_max_seqlen_q: int = 0
    cross_max_seqlen_k: int = 0
    cross_block_tables: torch.Tensor | None = None
    # Decode: context lens = encoder sequence lengths per request
    cross_context_lens: torch.Tensor | None = None
    cross_max_context_len: int = 0

    # --- Compilation / CUDA-graph fields (mirror vLLM ForwardContext) ---
    # Maps layer prefix -> live nn.Module for custom-op runtime lookup.
    no_compile_layers: dict[str, "nn.Module"] = field(default_factory=dict)
    # Runtime mode for CUDAGraphWrapper dispatch.
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    # Batch size key used by CUDAGraphWrapper for per-shape graph caching.
    batch_size_for_graph: int = 0


# Global module registry populated once at model init; copied into each
# Context so compiled custom ops can resolve their target modules.
_STATIC_NO_COMPILE_LAYERS: dict[str, "nn.Module"] = {}


def register_no_compile_layers(layers: dict[str, "nn.Module"]) -> None:
    """Register attention/MoE modules for custom-op lookup during compiled
    execution.  Called once after model construction."""
    _STATIC_NO_COMPILE_LAYERS.update(layers)


def get_no_compile_layers() -> dict[str, "nn.Module"]:
    return _STATIC_NO_COMPILE_LAYERS


def auto_register_no_compile_layers(model: "nn.Module") -> None:
    """Walk *model* and register every MoE and Attention sub-module by its
    fully-qualified prefix so custom ops can find them at runtime.

    Recognized types (by class name to avoid circular imports):
      - ``Qwen3MoE``, ``MixtralMoE``, ``DeepSeekMoE``  (MoE blocks)
      - ``Attention``                                    (paged-KV attention impl)

    Also sets ``_layer_name`` on each module so it knows its own key.
    ``_use_custom_op`` remains ``False`` until compilation is enabled.
    """
    _TARGET_NAMES = {"Qwen3MoE", "MixtralMoE", "GptOssMoE", "DeepSeekMoE", "Attention"}
    layers: dict[str, "nn.Module"] = {}
    for name, mod in model.named_modules():
        if type(mod).__name__ in _TARGET_NAMES:
            layers[name] = mod
            mod._layer_name = name  # type: ignore[attr-defined]
    register_no_compile_layers(layers)


def enable_custom_ops() -> None:
    """Switch all registered no-compile layers to dispatch through custom ops.
    Called once after torch.compile is applied to the model."""
    for mod in _STATIC_NO_COMPILE_LAYERS.values():
        mod._use_custom_op = True  # type: ignore[attr-defined]


def disable_custom_ops() -> None:
    """Revert to eager dispatch (used for testing/fallback)."""
    for mod in _STATIC_NO_COMPILE_LAYERS.values():
        mod._use_custom_op = False  # type: ignore[attr-defined]


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None,
                context_lens=None, block_tables=None,
                max_context_len=0, chunked_context=None,
                req_id_per_token=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k, slot_mapping,
                       context_lens, block_tables, max_context_len,
                       chunked_context=chunked_context,
                       req_id_per_token=req_id_per_token,
                       no_compile_layers=_STATIC_NO_COMPILE_LAYERS)


def set_mixed_context(
    slot_mapping,
    num_prefill_tokens, num_decode_tokens, num_prefill_seqs,
    prefill_cu_seqlens_q, prefill_cu_seqlens_k,
    prefill_max_seqlen_q, prefill_max_seqlen_k,
    prefill_block_tables,
    decode_context_lens, decode_block_tables, decode_max_context_len,
    logit_indices,
    chunked_context=None,
    req_id_per_token=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=True, is_mixed=True,
        slot_mapping=slot_mapping,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        num_prefill_seqs=num_prefill_seqs,
        prefill_cu_seqlens_q=prefill_cu_seqlens_q,
        prefill_cu_seqlens_k=prefill_cu_seqlens_k,
        prefill_max_seqlen_q=prefill_max_seqlen_q,
        prefill_max_seqlen_k=prefill_max_seqlen_k,
        prefill_block_tables=prefill_block_tables,
        decode_context_lens=decode_context_lens,
        decode_block_tables=decode_block_tables,
        decode_max_context_len=decode_max_context_len,
        logit_indices=logit_indices,
        chunked_context=chunked_context,
        req_id_per_token=req_id_per_token,
        no_compile_layers=_STATIC_NO_COMPILE_LAYERS,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context(no_compile_layers=_STATIC_NO_COMPILE_LAYERS)


@contextmanager
def set_forward_context(
    is_prefill: bool = False,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_size_for_graph: int = 0,
    **ctx_kwargs,
):
    """Context manager that sets both KV-cache metadata and compile/graph
    fields for the duration of a forward pass.

    ``no_compile_layers`` is always populated from the global registry so
    custom ops can resolve modules without the caller threading it through.
    """
    global _CONTEXT
    prev = _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        no_compile_layers=_STATIC_NO_COMPILE_LAYERS,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_size_for_graph=batch_size_for_graph,
        **ctx_kwargs,
    )
    try:
        yield _CONTEXT
    finally:
        _CONTEXT = prev
