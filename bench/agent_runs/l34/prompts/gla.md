I need you to write a high-performance replacement for the following PyTorch nn.Module operator used in an LLM inference engine. Your goal is to produce a kernel that is FASTER than the baseline implementation while remaining numerically correct.

## Baseline implementation

```python
"""Standalone GLA (Gated Linear Attention) model implementation.

Matches FLA checkpoint weight names exactly:
  model.embeddings.weight
  model.layers.{i}.attn_norm.weight
  model.layers.{i}.attn.{q,k,v,g,o}_proj.weight
  model.layers.{i}.attn.gk_proj.{0,1}.{weight,bias}
  model.layers.{i}.attn.g_norm_swish_gate.weight
  model.layers.{i}.mlp_norm.weight
  model.layers.{i}.mlp.{gate,up,down}_proj.weight
  model.norm.weight
  lm_head.weight
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.embedding import Embedding
from ..L1.linear import Linear
from ..L1.rms_norm import RMSNorm
from ..L3.gla_decoder import GLADecoderLayer
from .recurrent_cache import CausalLMOutputWithPast, RecurrentCache


@dataclass
class GLAConfig:
    hidden_size: int = 2560
    num_heads: int = 5
    num_hidden_layers: int = 32
    vocab_size: int = 32000
    expand_k: float = 0.5
    expand_v: float = 1.0
    hidden_ratio: int = 4
    intermediate_size: int | None = None
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.intermediate_size is None:
            intermediate = int(self.hidden_size * self.hidden_ratio * 2 / 3)
            self.intermediate_size = 256 * ((intermediate + 255) // 256)

    @classmethod
    def from_dict(cls, data: dict) -> "GLAConfig":
        keys = {f.name for f in cls.__dataclass_fields__.values() if f.name != "dtype"}
        kwargs = {k: data[k] for k in keys if k in data}
        return cls(**kwargs)

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "GLAConfig":
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class GLAModel(nn.Module):
    def __init__(self, config: GLAConfig):
        super().__init__()
        self.embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GLADecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: RecurrentCache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, RecurrentCache | None]:
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds
        if use_cache and past_key_values is None:
            past_key_values = RecurrentCache()

        for layer in self.layers:
            hidden_states, _, past_key_values = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        norm_dtype = self.norm.weight.dtype
        hidden_states = self.norm(
            hidden_states.to(dtype=norm_dtype).reshape(-1, hidden_states.size(-1))
        ).reshape_as(hidden_states)
        return hidden_states, past_key_values


class GLAForCausalLM(nn.Module):
    def __init__(self, config: GLAConfig):
        super().__init__()
        self.config = config
        self.model = GLAModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.emb.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: RecurrentCache | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        num_logits_to_keep: int = 0,
        logits_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        # Generation-time fast path: cap lm_head + fp32 upcast to the
        # last ``num_logits_to_keep`` positions. Saves several GB of
        # transient fp32 logits at large prefill batch sizes.
        if logits_indices is not None:
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            hidden_states = hidden_states.index_select(0, logits_indices).unsqueeze(1)
        elif num_logits_to_keep > 0:
            hidden_states = hidden_states[:, -num_logits_to_keep:, :]
        logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return CausalLMOutputWithPast(
            logits=logits, past_key_values=past_key_values, loss=loss,
        )

```

## Requirements

1. Your replacement class MUST:
   - Be named `GLAForCausalLM` (exactly)
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
- The class definition for `GLAForCausalLM`
- Any helper functions, Triton kernels, or CUDA source strings needed

Do NOT include any explanation outside the code block. Do NOT include if __name__ == '__main__' blocks or test code.