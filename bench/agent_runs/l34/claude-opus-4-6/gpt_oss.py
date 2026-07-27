from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L4.gpt_oss import GptOssConfig
from fastkernels.tasks.baseline.L1.yarn_rotary_emb import YaRNRotaryEmbedding
from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.gpt_oss_moe import GptOssMoE


@triton.jit
def _rms_norm_kernel(
    X, W, Out, stride,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    mask = off < N
    x = tl.load(X + row * stride + off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + off, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    tl.store(Out + row * stride + off, x * tl.math.rsqrt(var + eps) * w, mask=mask)


@triton.jit
def _fused_add_rms_norm_kernel(
    X, R, W, stride,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    mask = off < N
    x = tl.load(X + row * stride + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + row * stride + off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + off, mask=mask, other=0.0).to(tl.float32)
    s = x + r
    tl.store(R + row * stride + off, s, mask=mask)
    var = tl.sum(s * s, axis=0) / N
    tl.store(X + row * stride + off, s * tl.math.rsqrt(var + eps) * w, mask=mask)


class _TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self._block = triton.next_power_of_2(hidden_size)
        self._nw = 8

    @staticmethod
    def _native(x, weight, eps, hidden_size, residual=None):
        orig = x.dtype
        x = x.float()
        if residual is not None:
            x = x + residual.float()
            residual = x.to(orig)
        v = x.pow(2).mean(-1, keepdim=True)
        x = (x * torch.rsqrt(v + eps)).to(orig)
        if weight is not None:
            x = x * weight
        return x if residual is None else (x, residual)

    def forward(self, x, residual=None):
        if torch.compiler.is_compiling():
            return self._native(x, self.weight, self.eps, self.hidden_size, residual)
        shape = x.shape
        x2 = x.view(-1, self.hidden_size)
        M = x2.shape[0]
        w = self.weight
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
        if residual is not None:
            r2 = residual.view(-1, self.hidden_size)
            _fused_add_rms_norm_kernel[(M,)](
                x2, r2, w, x2.stride(0),
                N=self.hidden_size, eps=self.eps, BLOCK=self._block,
                num_warps=self._nw,
            )
            return x.view(shape), residual
        out = torch.empty_like(x2)
        _rms_norm_kernel[(M,)](
            x2, w, out, x2.stride(0),
            N=self.hidden_size, eps=self.eps, BLOCK=self._block,
            num_warps=self._nw,
        )
        return out.view(shape)


class _DecoderLayer(nn.Module):
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
        self.input_layernorm = _TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual, rotary_emb):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, rotary_emb=rotary_emb)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class _Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            _DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = _TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        model_dtype = getattr(config, "dtype", torch.bfloat16)
        self.rotary_emb.register_buffer(
            "cos_sin_cache",
            self.rotary_emb.cos_sin_cache.to(model_dtype),
            persistent=False,
        )

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        rotary = self.rotary_emb
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, rotary)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class GptOssForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config
        self.model = _Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits
