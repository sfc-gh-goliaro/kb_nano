"""FP8 KV cache store and gather for MLA (656 bytes/token).

Uses vLLM's CUDA kernels for high-performance FP8 MLA cache operations.

Cache format per token (656 bytes):

  * ``[0:512]`` — ``kv_c_normed`` as FP8 (``float8_e4m3fn``).
  * ``[512:528]`` — four per-group FP32 UE8M0 scales (128 dims per group).
  * ``[528:656]`` — ``k_pe`` as 64 ``bfloat16`` values (128 bytes).

Cache tensor shape: ``[num_blocks, block_size, 656]`` with ``dtype=torch.uint8``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import vllm._C  # noqa: F401  — registers torch.ops._C_cache_ops

_BYTES_PER_TOKEN = 656
_KV_C_DIM = 512
_K_PE_DIM = 64


class StoreKVCacheFP8MLA(nn.Module):
    """Store ``kv_c_normed`` and ``k_pe`` into FP8 MLA paged cache.

    Wraps ``torch.ops._C_cache_ops.concat_and_cache_mla`` with
    ``kv_cache_dtype="fp8_ds_mla"`` which fuses per-block UE8M0
    FP8 quantization of kv_c_normed with BF16 k_pe storage.

    Args:
        kv_c_normed: ``[N, 512]`` BF16 — compressed KV after layernorm.
        k_pe: ``[N, 1, 64]`` or ``[N, 64]`` BF16 — RoPE key component.
        kv_cache: ``[num_blocks, block_size, 656]`` uint8.
        slot_mapping: ``[N]`` int64 — linear slot index per token (``-1`` skips).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "_k_scale", torch.zeros(1, dtype=torch.float32), persistent=False,
        )

    def forward(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        k_pe_2d = k_pe.reshape(k_pe.shape[0], -1)
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c_normed, k_pe_2d, kv_cache, slot_mapping,
            "fp8_ds_mla", self._k_scale,
        )


class GatherKVCacheFP8MLA(nn.Module):
    """Gather and upconvert KV from FP8 MLA paged cache to BF16.

    Wraps ``torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache``
    which gathers FP8-quantized kv_c_normed and BF16 k_pe from paged cache,
    dequantizes the FP8 portion, and writes the result as a contiguous
    BF16 workspace tensor.

    Returns:
        ``workspace``: ``[total_tokens, 576]`` BF16 — dequantized kv_c_normed
        (512 dims) concatenated with k_pe (64 dims).
    """

    def forward(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace_starts: torch.Tensor,
        num_seqs: int,
        workspace: torch.Tensor,
    ) -> None:
        torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache(
            kv_cache, workspace, block_table, seq_lens,
            workspace_starts, num_seqs,
        )
