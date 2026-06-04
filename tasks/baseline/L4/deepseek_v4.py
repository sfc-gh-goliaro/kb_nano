"""DeepSeek V4-Flash model implementation.

Supports MLA with sliding-window + compression, MoE with hash/noaux_tc routing,
MXFP4/FP8 expert weights, and Multi-Head Cache (MHC) residual connections.
Uses YARN-scaled RoPE, FP8 quantization, and tensor parallelism.

Reference: vllm/model_executor/models/deepseek_v4.py (vLLM 0.20.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L1.rms_norm import RMSNorm
from ..L1.yarn_rotary_emb import YarnRotaryEmbedding
from ..L3.deepseek_v4_decoder import DeepSeekV4DecoderLayer


@dataclass
class DeepSeekV4Config:
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    vocab_size: int = 129280
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # MLA params
    head_dim: int = 512
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    qk_rope_head_dim: int = 64
    sliding_window: int = 128

    # MoE params
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 1.5
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0
    num_hash_layers: int = 3

    # MHC params
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20

    # DSA params
    index_topk: int = 512
    index_n_heads: int = 64
    index_head_dim: int = 128

    # Compression
    compress_ratios: list = field(default_factory=list)
    compress_rope_theta: float = 160000.0

    # YARN RoPE params
    rope_parameters: dict = field(default_factory=lambda: {
        'rope_type': 'yarn',
        'factor': 16.0,
        'mscale': 0,
        'mscale_all_dim': 0,
        'attn_factor': 1.0,
        'beta_fast': 32,
        'beta_slow': 1,
        'original_max_position_embeddings': 65536,
    })

    # Quantization
    quantization_config: dict = field(default_factory=lambda: {
        'quant_method': 'fp8',
        'weight_block_size': [128, 128],
        'scale_fmt': 'ue8m0',
    })
    expert_dtype: str = "fp4"

    # Number of MTP layers
    num_nextn_predict_layers: int = 1

    model_type: str = "deepseek_v4"
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_name: str) -> "DeepSeekV4Config":
        import json
        import os
        try:
            hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except (ValueError, KeyError):
            from huggingface_hub import hf_hub_download
            if os.path.isdir(model_name):
                path = os.path.join(model_name, "config.json")
            else:
                path = hf_hub_download(model_name, "config.json")
            with open(path) as f:
                cfg = json.load(f)
            from types import SimpleNamespace
            hf = SimpleNamespace(**cfg)

        rope = getattr(hf, 'rope_scaling', {}) or {}
        rope_params = {
            'rope_type': rope.get('type', rope.get('rope_type', 'yarn')),
            'factor': rope.get('factor', 16.0),
            'mscale': 0,
            'mscale_all_dim': 0,
            'attn_factor': rope.get('attn_factor', 1.0),
            'beta_fast': rope.get('beta_fast', 32),
            'beta_slow': rope.get('beta_slow', 1),
            'original_max_position_embeddings': rope.get(
                'original_max_position_embeddings', 65536),
        }

        quant_cfg = getattr(hf, 'quantization_config', {}) or {}

        compress_ratios = getattr(hf, 'compress_ratios', [])

        return cls(
            hidden_size=hf.hidden_size,
            moe_intermediate_size=getattr(hf, 'moe_intermediate_size', 2048),
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=getattr(hf, 'num_key_value_heads', 1),
            vocab_size=hf.vocab_size,
            max_position_embeddings=hf.max_position_embeddings,
            rms_norm_eps=getattr(hf, 'rms_norm_eps', 1e-6),
            rope_theta=getattr(hf, 'rope_theta', 10000.0),
            head_dim=getattr(hf, 'head_dim', 512),
            q_lora_rank=getattr(hf, 'q_lora_rank', 1024),
            o_lora_rank=getattr(hf, 'o_lora_rank', 1024),
            o_groups=getattr(hf, 'o_groups', 8),
            qk_rope_head_dim=getattr(hf, 'qk_rope_head_dim', 64),
            sliding_window=getattr(hf, 'sliding_window', 128),
            n_routed_experts=getattr(hf, 'n_routed_experts', 256),
            n_shared_experts=getattr(hf, 'n_shared_experts', 1),
            num_experts_per_tok=getattr(hf, 'num_experts_per_tok', 6),
            routed_scaling_factor=getattr(hf, 'routed_scaling_factor', 1.5),
            scoring_func=getattr(hf, 'scoring_func', 'sqrtsoftplus'),
            topk_method=getattr(hf, 'topk_method', 'noaux_tc'),
            norm_topk_prob=getattr(hf, 'norm_topk_prob', True),
            hidden_act=getattr(hf, 'hidden_act', 'silu'),
            swiglu_limit=getattr(hf, 'swiglu_limit', 10.0),
            num_hash_layers=getattr(hf, 'num_hash_layers', 3),
            hc_mult=getattr(hf, 'hc_mult', 4),
            hc_eps=getattr(hf, 'hc_eps', 1e-6),
            hc_sinkhorn_iters=getattr(hf, 'hc_sinkhorn_iters', 20),
            index_topk=getattr(hf, 'index_topk', 512),
            index_n_heads=getattr(hf, 'index_n_heads', 64),
            index_head_dim=getattr(hf, 'index_head_dim', 128),
            compress_ratios=compress_ratios,
            compress_rope_theta=getattr(hf, 'compress_rope_theta', 160000.0),
            rope_parameters=rope_params,
            quantization_config=quant_cfg,
            expert_dtype=getattr(hf, 'expert_dtype', 'fp4'),
            num_nextn_predict_layers=getattr(hf, 'num_nextn_predict_layers', 1),
        )


@torch.compile(backend="inductor")
def hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Final MHC head: combines hc_mult channels into 1.

    Matches vllm/model_executor/models/deepseek_v4.py:hc_head
    """
    x = hidden_states
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


class DeepSeekV4Model(nn.Module):
    def __init__(self, config: DeepSeekV4Config, quant_config: dict | None = None):
        super().__init__()
        self.config = config
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        hc_dim = config.hc_mult * config.hidden_size
        mix_hc = (2 + config.hc_mult) * config.hc_mult

        self.rotary_emb = YarnRotaryEmbedding(
            head_dim=config.qk_rope_head_dim,
            max_position_embeddings=config.rope_parameters.get(
                'original_max_position_embeddings', config.max_position_embeddings),
            rope_theta=config.rope_theta,
            scaling_factor=config.rope_parameters.get('factor', 1.0),
            attn_factor=config.rope_parameters.get('attn_factor', 1.0),
            beta_fast=config.rope_parameters.get('beta_fast', 32),
            beta_slow=config.rope_parameters.get('beta_slow', 1),
            mscale=0.0,
            mscale_all_dim=0.0,
            is_neox_style=False,
        )

        # Pre-allocate topk_indices_buffer for DSA indexer
        max_batched = getattr(config, 'max_num_batched_tokens', 16384)
        self.topk_indices_buffer = torch.empty(
            max_batched, config.index_topk, dtype=torch.int32,
        )

        self.layers = nn.ModuleList([
            DeepSeekV4DecoderLayer(
                config, layer_idx=i,
                rotary_emb=self.rotary_emb,
                quant_config=quant_config,
                topk_indices_buffer=self.topk_indices_buffer,
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # hc_head parameters
        self.hc_head_fn = nn.Parameter(
            torch.empty(config.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(config.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        # Expand to MHC shape: (batch, hc_mult, hidden_size)
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, input_ids)

        # hc_head: combine hc_mult channels
        hidden_states = hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class DeepSeekV4ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "w1": ("gate_up_proj", 0),
        "w3": ("gate_up_proj", 1),
        "wq_a": ("fused_wqa_wkv", 0),
        "wkv": ("fused_wqa_wkv", 1),
    }

    def __init__(self, config: DeepSeekV4Config, quant_config: dict | None = None):
        super().__init__()
        self.config = config
        self.model = DeepSeekV4Model(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

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
        result = self.lm_head.gather_greedy(partial_logits.float())
        return result
