I need you to write a high-performance replacement for the following PyTorch nn.Module operator used in an LLM inference engine. Your goal is to produce a kernel that is FASTER than the baseline implementation while remaining numerically correct.

## Baseline implementation

```python
"""GPT-OSS decoder layer: attention + MoE with RMSNorm residual connections.

Uses the shared ``LlamaAttention`` with ``use_sinks=True`` and
``sliding_window`` to implement GPT-OSS attention sinks and per-layer
sliding window. Rotary embedding is passed through forward (created
once at the model level and shared across layers).
"""

from __future__ import annotations

import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.attention import LlamaAttention
from ..L2.gpt_oss_moe import GptOssMoE


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            bias=True,
            o_proj_bias=True,
            use_sinks=True,
            sliding_window=config.sliding_window,
            layer_idx=layer_idx,
        )
        self.mlp = GptOssMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual, rotary_emb):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, rotary_emb=rotary_emb)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

```

## Requirements

1. Your replacement class MUST:
   - Be named `GptOssDecoderLayer` (exactly)
   - Subclass `torch.nn.Module`
   - Have the EXACT same `forward` signature (same parameter names, types, defaults)
   - Produce numerically equivalent outputs (or very close)
   - The `__init__` method is optional -- you only need to override it if you need to change initialization logic (e.g. pre-allocate buffers). If you do override it, keep the same signature.
2. You may use Triton, PyTorch, raw CUDA, or any combination. Aim for the highest performance possible on NVIDIA H200 GPUs.
3. Do NOT import `vllm`, `sglang`, or `sgl_kernel`.
3b. Your file is imported STANDALONE, outside the package, so RELATIVE imports (`from ..L1.x import Y`) will fail with ImportError. To reuse a baseline component, import it absolutely as `from fastkernels.tasks.baseline.L<n>.<module> import <Class>` (e.g. `from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm`), or inline the code you need. Never emit a relative import.
4. You may import `torch`, `triton`, `triton.language`, standard library modules, `flash_attn`, or JIT-compile CUDA. For inline CUDA strings use `torch.utils.cpp_extension.load_inline(name=..., cpp_sources=..., cuda_sources=..., functions=[...], build_directory='/mnt/weka/home/hao.zhang/async_rl_bench/kb_nano/agent/_cuda_build_cache/<unique_name>')`. Do NOT pass `cuda_sources` to `torch.utils.cpp_extension.load()` (it only takes file paths via `sources`).
5. Focus on PERFORMANCE: minimize memory traffic, maximize GPU occupancy, fuse operations where possible, and use vectorized loads/stores.

## Response format

Return ONLY a single Python code block (```python ... ```) containing:
- All necessary imports at the top
- The class definition for `GptOssDecoderLayer`
- Any helper functions, Triton kernels, or CUDA source strings needed

Do NOT include any explanation outside the code block. Do NOT include if __name__ == '__main__' blocks or test code.