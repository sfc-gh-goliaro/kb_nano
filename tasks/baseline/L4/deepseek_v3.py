"""DeepSeek V3 / V3.2 model implementation.

Uses MLA attention, MoE with grouped sigmoid routing, FP8,
and supports DP+EP parallelism.
V3.2 adds DeepSeek Sparse Attention (DSA) with FlashMLA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L1.rms_norm import RMSNorm
from ..L1.yarn_rotary_emb import YaRNRotaryEmbedding, yarn_get_mscale
from ..L3.deepseek_decoder import DeepSeekDecoderLayer


@dataclass
class DeepSeekV3Config:
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    vocab_size: int = 129280
    max_position_embeddings: int = 163840
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # YaRN RoPE
    rope_scaling_factor: float = 40.0
    rope_original_max_position_embeddings: int = 4096
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0

    # MLA dimensions
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512

    # MoE
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 3
    moe_layer_freq: int = 1
    n_group: int = 8
    topk_group: int = 4
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True

    tie_word_embeddings: bool = False
    dtype: torch.dtype = torch.bfloat16

    # Computed from YaRN mscale for attention scaling
    attn_scaling: float = 1.0

    # DSA Indexer (V3.2 only, None for V3)
    index_topk: int | None = None
    index_n_heads: int = 64
    index_head_dim: int = 128

    @classmethod
    def from_pretrained(cls, model_name: str) -> "DeepSeekV3Config":
        import json
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(model_name, "config.json")
        with open(cfg_path) as f:
            hf = json.load(f)
        rope = hf.get("rope_scaling", None) or {}

        cfg = cls(
            hidden_size=hf["hidden_size"],
            intermediate_size=hf["intermediate_size"],
            moe_intermediate_size=hf.get("moe_intermediate_size", 2048),
            num_hidden_layers=hf["num_hidden_layers"],
            num_attention_heads=hf["num_attention_heads"],
            num_key_value_heads=hf.get("num_key_value_heads", hf["num_attention_heads"]),
            vocab_size=hf["vocab_size"],
            max_position_embeddings=hf["max_position_embeddings"],
            rms_norm_eps=hf.get("rms_norm_eps", 1e-6),
            rope_theta=hf.get("rope_theta", 10000.0),
            rope_scaling_factor=rope.get("factor", 1.0),
            rope_original_max_position_embeddings=rope.get("original_max_position_embeddings", 4096),
            rope_beta_fast=rope.get("beta_fast", 32),
            rope_beta_slow=rope.get("beta_slow", 1),
            rope_mscale=rope.get("mscale", 1.0),
            rope_mscale_all_dim=rope.get("mscale_all_dim", 0.0),
            qk_nope_head_dim=hf.get("qk_nope_head_dim", 128),
            qk_rope_head_dim=hf.get("qk_rope_head_dim", 64),
            v_head_dim=hf.get("v_head_dim", 128),
            q_lora_rank=hf.get("q_lora_rank", 1536),
            kv_lora_rank=hf.get("kv_lora_rank", 512),
            n_routed_experts=hf.get("n_routed_experts", 256),
            n_shared_experts=hf.get("n_shared_experts", 1),
            num_experts_per_tok=hf.get("num_experts_per_tok", 8),
            first_k_dense_replace=hf.get("first_k_dense_replace", 3),
            moe_layer_freq=hf.get("moe_layer_freq", 1),
            n_group=hf.get("n_group", 8),
            topk_group=hf.get("topk_group", 4),
            topk_method=hf.get("topk_method", "noaux_tc"),
            scoring_func=hf.get("scoring_func", "sigmoid"),
            routed_scaling_factor=hf.get("routed_scaling_factor", 2.5),
            norm_topk_prob=hf.get("norm_topk_prob", True),
            tie_word_embeddings=hf.get("tie_word_embeddings", False),
            index_topk=hf.get("index_topk", None),
            index_n_heads=hf.get("index_n_heads", 64),
            index_head_dim=hf.get("index_head_dim", 128),
        )

        if cfg.rope_scaling_factor > 1.0:
            mscale = yarn_get_mscale(cfg.rope_scaling_factor, cfg.rope_mscale_all_dim)
            cfg.attn_scaling = mscale * mscale
        else:
            cfg.attn_scaling = 1.0

        return cfg


class DeepSeekV3Model(nn.Module):
    def __init__(self, config: DeepSeekV3Config, quant_config: dict | None = None):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = YaRNRotaryEmbedding(
            config.qk_rope_head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            scaling_factor=config.rope_scaling_factor,
            original_max_position_embeddings=config.rope_original_max_position_embeddings,
            beta_fast=config.rope_beta_fast,
            beta_slow=config.rope_beta_slow,
            mscale=config.rope_mscale,
            mscale_all_dim=config.rope_mscale_all_dim,
        )

        self.is_v32 = config.index_topk is not None

        topk_indices_buffer = None
        indexer_rotary_emb = None

        if self.is_v32:
            topk_indices_buffer = torch.empty(
                16384,
                config.index_topk,
                dtype=torch.int32,
            )
            indexer_rotary_emb = YaRNRotaryEmbedding(
                config.qk_rope_head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                scaling_factor=config.rope_scaling_factor,
                original_max_position_embeddings=config.rope_original_max_position_embeddings,
                beta_fast=config.rope_beta_fast,
                beta_slow=config.rope_beta_slow,
                mscale=config.rope_mscale,
                mscale_all_dim=config.rope_mscale_all_dim,
                is_neox=True,
            )

        self.topk_indices_buffer = topk_indices_buffer
        self.indexer_rotary_emb = indexer_rotary_emb

        self.layers = nn.ModuleList([
            DeepSeekDecoderLayer(
                config, layer_idx=i, rotary_emb=self.rotary_emb,
                quant_config=quant_config,
                topk_indices_buffer=topk_indices_buffer,
                indexer_rotary_emb=indexer_rotary_emb,
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        from ..L2.deepseek_moe import set_ep_max_n, _ep_cached_max_n
        from ....infra.tp import _ep_size, get_ep_group
        ep = _ep_size()
        owned_max_n = False
        if ep > 1 and _ep_cached_max_n is None:
            import torch.distributed as dist
            ep_group = get_ep_group()
            local_n = input_ids.size(0)
            max_n_t = torch.tensor([local_n], dtype=torch.int64,
                                   device=input_ids.device)
            dist.all_reduce(max_n_t, op=dist.ReduceOp.MAX, group=ep_group)
            set_ep_max_n(int(max_n_t.item()))
            owned_max_n = True

        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if owned_max_n:
            set_ep_max_n(None)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepSeekV3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: DeepSeekV3Config, quant_config: dict | None = None):
        super().__init__()
        self.config = config
        self.model = DeepSeekV3Model(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def forward_with_lm_proj(self, input_ids, positions):
        hidden_states = self.model(input_ids, positions)
        return self.lm_head.project(hidden_states)

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
