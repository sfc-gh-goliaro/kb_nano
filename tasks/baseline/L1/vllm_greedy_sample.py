"""vLLM-compatible greedy sampling helpers.

vLLM's GPU sampler uses the Gumbel sampler kernel even for temperature=0.
That matters for GPT-OSS because many logits are quantized onto exact ties:
the kernel first reduces 1024-token blocks, then PyTorch selects the first
max block.  Plain ``torch.argmax`` or local TP max-pair reduction has different
tie behavior.
"""

from __future__ import annotations

import torch

from vllm.v1.worker.gpu.sample.gumbel import _gumbel_sample_kernel
from vllm.triton_utils import triton


_BLOCK_SIZE = 1024


def vllm_greedy_sample(
    logits: torch.Tensor,
    *,
    local_argmax: torch.Tensor | None = None,
    local_max: torch.Tensor | None = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    temperature: torch.Tensor | None = None,
    seeds: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return greedy token ids with vLLM 0.22 sampler tie semantics."""

    if not logits.is_cuda:
        return logits.argmax(dim=-1).view(-1)

    num_tokens, vocab_size = logits.shape
    num_blocks = triton.cdiv(vocab_size, _BLOCK_SIZE)
    device = logits.device

    if local_argmax is None:
        local_argmax = torch.empty(
            num_tokens, num_blocks, dtype=torch.int64, device=device,
        )
    else:
        local_argmax = local_argmax[:num_tokens, :num_blocks]
    if local_max is None:
        local_max = torch.empty(
            num_tokens, num_blocks, dtype=torch.float32, device=device,
        )
    else:
        local_max = local_max[:num_tokens, :num_blocks]

    if expanded_idx_mapping is None:
        expanded_idx_mapping = torch.zeros(
            num_tokens, dtype=torch.int32, device=device,
        )
    else:
        expanded_idx_mapping = expanded_idx_mapping[:num_tokens]
    if temperature is None:
        temperature = torch.zeros(max(num_tokens, 1), dtype=torch.float32,
                                  device=device)
    if seeds is None:
        seeds = torch.zeros(max(num_tokens, 1), dtype=torch.int64,
                            device=device)
    if positions is None:
        positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    else:
        positions = positions[:num_tokens]

    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        None,
        0,
        None,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seeds,
        positions,
        temperature,
        vocab_size,
        BLOCK_SIZE=_BLOCK_SIZE,
        APPLY_TEMPERATURE=False,
        USE_FP64=False,
    )
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    return local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
