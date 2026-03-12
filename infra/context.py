"""Global inference context for paged KV cache coordination."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


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


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None,
                context_lens=None, block_tables=None,
                max_context_len=0):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k, slot_mapping,
                       context_lens, block_tables, max_context_len)


def set_mixed_context(
    slot_mapping,
    num_prefill_tokens, num_decode_tokens, num_prefill_seqs,
    prefill_cu_seqlens_q, prefill_cu_seqlens_k,
    prefill_max_seqlen_q, prefill_max_seqlen_k,
    prefill_block_tables,
    decode_context_lens, decode_block_tables, decode_max_context_len,
    logit_indices,
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
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
