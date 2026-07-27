from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/fastkernels_torch_extensions")

import torch
import torch.nn as nn
from transformers import AutoConfig

from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from fastkernels.tasks.baseline.L3.llama_decoder import LlamaDecoderLayer


@dataclass
class LlamaConfig:
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 128256
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling_factor: float = 8.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_max_position_embeddings: int = 8192
    dtype: torch.dtype = torch.bfloat16
    qkv_bias: bool = False

    @classmethod
    def from_pretrained(cls, model_name: str) -> "LlamaConfig":
        hf = AutoConfig.from_pretrained(model_name)
        rope_params = getattr(hf, "rope_parameters", None) or {}
        rope = getattr(hf, "rope_scaling", None) or {}
        rope = {**rope, **rope_params}
        rope_theta = rope.get("rope_theta") or getattr(hf, "rope_theta", 500000.0)
        is_qwen2 = getattr(hf, "model_type", "") in ("qwen2", "qwen2_moe")
        return cls(
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads),
            vocab_size=hf.vocab_size,
            max_position_embeddings=hf.max_position_embeddings,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling_factor=rope.get("factor", 1.0),
            rope_low_freq_factor=rope.get("low_freq_factor", 1.0),
            rope_high_freq_factor=rope.get("high_freq_factor", 1.0),
            rope_original_max_position_embeddings=rope.get(
                "original_max_position_embeddings", hf.max_position_embeddings,
            ),
            qkv_bias=is_qwen2,
        )


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            rope_scaling_factor=config.rope_scaling_factor,
            rope_low_freq_factor=config.rope_low_freq_factor,
            rope_high_freq_factor=config.rope_high_freq_factor,
            rope_original_max_position_embeddings=config.rope_original_max_position_embeddings,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, rotary_emb=self.rotary_emb, bias=config.qkv_bias)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.capture_aux_hidden_states: bool = False
        self._aux_layer_ids: list[int] = []
        self._aux_layer_set: set[int] = set()

    @property
    def aux_layer_ids(self) -> list[int]:
        return self._aux_layer_ids

    @aux_layer_ids.setter
    def aux_layer_ids(self, value: list[int]):
        self._aux_layer_ids = list(value)
        self._aux_layer_set = set(value)

    def forward(self, input_ids, positions, inputs_embeds=None):
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        residual = None

        if not self.capture_aux_hidden_states:
            for layer in self.layers:
                hidden_states, residual = layer.forward(positions, hidden_states, residual)
            hidden_states, _ = self.norm.forward(hidden_states, residual)
            return hidden_states

        aux_hidden_states: list[torch.Tensor] = []
        aux_set = self._aux_layer_set
        for i, layer in enumerate(self.layers):
            if i in aux_set:
                aux_hidden_states.append(
                    hidden_states if residual is None else hidden_states + residual
                )
            hidden_states, residual = layer.forward(positions, hidden_states, residual)
        hidden_states, _ = self.norm.forward(hidden_states, residual)
        return hidden_states, aux_hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self._graph = None
        self._graph_key = None
        self._graph_output = None
        self._graph_calls = 0
        self._graph_disabled = False

    def _can_use_graph(self, input_ids, positions):
        return (
            not self._graph_disabled
            and not self.training
            and not torch.is_grad_enabled()
            and isinstance(input_ids, torch.Tensor)
            and isinstance(positions, torch.Tensor)
            and input_ids.is_cuda
            and positions.is_cuda
            and not self.model.capture_aux_hidden_states
        )

    @staticmethod
    def _tensor_key(x):
        return (x.data_ptr(), tuple(x.shape), tuple(x.stride()), x.dtype, x.device.index)

    def _eager_forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def forward(self, input_ids, positions):
        if not self._can_use_graph(input_ids, positions):
            return self._eager_forward(input_ids, positions)

        key = (self._tensor_key(input_ids), self._tensor_key(positions))
        if self._graph is not None and self._graph_key == key:
            self._graph.replay()
            return self._graph_output

        self._graph_calls += 1
        if self._graph_calls == 1:
            return self._eager_forward(input_ids, positions)

        try:
            stream = torch.cuda.current_stream(input_ids.device)
            stream.synchronize()
            warmup_out = self._eager_forward(input_ids, positions)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_out = self._eager_forward(input_ids, positions)
            self._graph = graph
            self._graph_key = key
            self._graph_output = graph_out
            return graph_out
        except Exception:
            self._graph = None
            self._graph_key = None
            self._graph_output = None
            self._graph_disabled = True
            return warmup_out if "warmup_out" in locals() else self._eager_forward(input_ids, positions)

    def forward_with_lm_proj(self, input_ids, positions):
        out = self.model(input_ids, positions)
        if isinstance(out, tuple):
            hidden_states = out[0]
        else:
            hidden_states = out
        return self.lm_head.project(hidden_states)

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None):
        num_layers = self.config.num_hidden_layers
        if layer_ids is None:
            layer_ids = [2, num_layers // 2, num_layers - 3]
        self.model.capture_aux_hidden_states = True
        self.model.aux_layer_ids = list(layer_ids)
        self._graph_disabled = True
        self._graph = None
        self._graph_output = None

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits

    def compute_logits_decode(self, partial_logits):
        logits = self.lm_head.gather_logits(partial_logits)
        if logits is not None:
            logits = logits.float()
        return logits

    def greedy_sample_decode(self, partial_logits):
        return self.lm_head.gather_greedy(partial_logits.float())
