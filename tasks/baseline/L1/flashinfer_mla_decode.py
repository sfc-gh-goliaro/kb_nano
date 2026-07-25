"""TRTLLM-gen MLA decode kernel (via FlashInfer, Blackwell only).

FlashMLA's dense decode kernel is compiled for SM90a only and raises
"Dense decode MLA is only supported on SM90a architecture" on Blackwell, so
sm100+ needs FlashInfer's TRTLLM-gen MLA decode instead. vLLM makes the same
switch: ``FLASHINFER_MLA`` (``v1/attention/backends/mla/flashinfer_mla.py``) is
gated on ``capability.major == 10`` and wraps the same entry point.

Accepts the same arguments as :class:`FlashMLADecode` minus FlashMLA's tile
scheduler metadata, which the TRTLLM kernel does not use.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

# vLLM sizes this at 128 MiB (FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE).
_WORKSPACE_BYTES = 128 * 1024 * 1024
_WORKSPACES: dict[tuple, torch.Tensor] = {}


def _workspace(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    buf = _WORKSPACES.get(key)
    if buf is None:
        buf = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _WORKSPACES[key] = buf
    return buf


class TRTLLMMLADecode(nn.Module):
    """Paged MLA decode on Blackwell.

    The FlashInfer kernel requires ``qk_nope_head_dim`` in (64, 128) and a KV
    cache page size of 32 or 64, matching vLLM's
    ``FlashInferMLABackend.supports_combination`` / ``get_supported_kernel_block_sizes``.
    """

    SUPPORTED_PAGE_SIZES = (32, 64)
    SUPPORTED_QK_NOPE = (64, 128)

    def __init__(self, qk_nope_head_dim: int, kv_lora_rank: int,
                 qk_rope_head_dim: int):
        super().__init__()
        self.qk_nope_head_dim = qk_nope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

    @classmethod
    def supported(cls, qk_nope_head_dim: int, page_size: int) -> bool:
        return (qk_nope_head_dim in cls.SUPPORTED_QK_NOPE
                and page_size in cls.SUPPORTED_PAGE_SIZES)

    @staticmethod
    def _pad_block_table(block_table, page_size: int):
        """Widen the block table to the page count the kernel demands.

        FlashInfer checks ``block_num % (128 / block_size) == 0`` on the page
        dimension (``_check_trtllm_gen_mla_shape``), so a table sized from
        ``max_model_len`` can be rejected outright -- Kimi-Linear at reduced scale
        produced 23 pages against a required multiple of 2. Padding with page 0 is
        harmless because ``seq_lens`` bounds how far the kernel reads.
        """
        mult = max(1, 128 // page_size)
        rem = block_table.shape[1] % mult
        if rem == 0:
            return block_table
        return torch.nn.functional.pad(block_table, (0, mult - rem), value=0)

    def forward(self, q, kv_cache, block_table, cache_seqlens,
                max_seq_len: int, softmax_scale: float, bmm2_scale: float = 1.0):
        """
        Args:
            q: ``[num_decode_tokens, num_heads, kv_lora_rank + qk_rope_head_dim]``
               latent-absorbed queries (one token per sequence).
            kv_cache: ``[num_blocks, page_size, kv_lora_rank + qk_rope_head_dim]``.
            block_table: ``[batch, max_blocks]`` int32.
            cache_seqlens: ``[batch]`` int32 context lengths.
            softmax_scale: becomes ``bmm1_scale``; for a BF16 cache the q/k
                dequant scales are 1.0, so this is the whole scale, exactly as
                vLLM computes ``q_scale * k_scale * scale``.

        Returns:
            ``[num_decode_tokens, num_heads, kv_lora_rank]``
        """
        block_table = self._pad_block_table(block_table, kv_cache.shape[1])
        o = trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=_workspace(q.device),
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_table,
            seq_lens=cache_seqlens,
            max_seq_len=max_seq_len,
            bmm1_scale=softmax_scale,
            bmm2_scale=bmm2_scale,
        )
        return o.reshape(-1, o.shape[-2], o.shape[-1])
