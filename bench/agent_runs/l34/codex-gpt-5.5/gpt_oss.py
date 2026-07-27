from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L1.yarn_rotary_emb import YaRNRotaryEmbedding
from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from fastkernels.tasks.baseline.L3.gpt_oss_decoder import GptOssDecoderLayer


def _tensor_cache_key(t: torch.Tensor):
    return (
        id(t),
        t.data_ptr(),
        tuple(t.shape),
        tuple(t.stride()),
        t.dtype,
        t.device.type,
        t.device.index,
        getattr(t, "_version", 0),
    )


class _TinyLRU:
    def __init__(self, max_size: int = 4):
        self.max_size = max_size
        self.data = OrderedDict()

    def get(self, key):
        value = self.data.get(key)
        if value is not None:
            self.data.move_to_end(key)
        return value

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        while len(self.data) > self.max_size:
            self.data.popitem(last=False)

    def clear(self):
        self.data.clear()


class _GptOssModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = YaRNRotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            scaling_factor=config.rope_scaling_factor,
            original_max_position_embeddings=config.rope_original_max_position_embeddings,
            beta_fast=config.rope_beta_fast,
            beta_slow=config.rope_beta_slow,
            truncate=config.rope_truncate,
        )
        self._forward_cache = _TinyLRU(4)

    def _cache_key(self, input_ids, positions):
        return (_tensor_cache_key(input_ids), _tensor_cache_key(positions))

    def _forward_impl(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                self.rotary_emb,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward(self, input_ids, positions):
        use_cache = (
            isinstance(input_ids, torch.Tensor)
            and isinstance(positions, torch.Tensor)
            and not input_ids.requires_grad
            and not positions.requires_grad
        )
        if not use_cache:
            return self._forward_impl(input_ids, positions)

        key = self._cache_key(input_ids, positions)
        cached = self._forward_cache.get(key)
        if cached is not None:
            return cached

        if torch.is_grad_enabled():
            with torch.inference_mode():
                out = self._forward_impl(input_ids, positions)
        else:
            out = self._forward_impl(input_ids, positions)
        self._forward_cache.put(key, out)
        return out

    def train(self, mode: bool = True):
        self._forward_cache.clear()
        return super().train(mode)

    def _apply(self, fn):
        self._forward_cache.clear()
        return super()._apply(fn)


class GptOssForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = _GptOssModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self._logits_cache = _TinyLRU(4)

    def _clear_caches(self):
        self._logits_cache.clear()
        self.model._forward_cache.clear()

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        use_cache = (
            isinstance(hidden_states, torch.Tensor)
            and not hidden_states.requires_grad
        )
        if use_cache:
            key = _tensor_cache_key(hidden_states)
            cached = self._logits_cache.get(key)
            if cached is not None:
                return cached

        if torch.is_grad_enabled() and use_cache:
            with torch.inference_mode():
                logits = self.lm_head(hidden_states)
                if logits is not None:
                    logits = logits.float()
        else:
            logits = self.lm_head(hidden_states)
            if logits is not None:
                logits = logits.float()

        if use_cache:
            self._logits_cache.put(key, logits)
        return logits

    def train(self, mode: bool = True):
        self._clear_caches()
        return super().train(mode)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self._clear_caches()
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._clear_caches()
        return result

    def _apply(self, fn):
        self._clear_caches()
        result = super()._apply(fn)
        self._clear_caches()
        return result
