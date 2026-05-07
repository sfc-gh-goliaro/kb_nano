"""Qwen3-VL-MoE model: vision encoder with DeepStack + Qwen3-MoE language model.

Combines Qwen3VisionTransformer (BF16) with a Qwen3-MoE language model
using 128 FP8 experts, top-8 routing, and moe_intermediate_size=768.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L1.mrope import MRotaryEmbedding
from ..L1.rms_norm import RMSNorm
from ..L1.mrope_input_positions import MRopeInputPositions
from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L3.qwen3_moe_decoder import Qwen3MoeDecoderLayer
from .qwen3_vl import Qwen3VLVisionConfig, Qwen3VisionTransformer


@dataclass
class Qwen3VLMoeConfig:
    hidden_size: int = 2048
    intermediate_size: int = 6144
    moe_intermediate_size: int = 768
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 151936
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    tie_word_embeddings: bool = False
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    mrope_interleaved: bool = True
    image_token_id: int = 151655
    video_token_id: int = 151656
    num_experts: int = 128
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    vision: Qwen3VLVisionConfig = field(default_factory=Qwen3VLVisionConfig)
    dtype: torch.dtype = torch.bfloat16
    fp8_block_size: tuple[int, int] | None = None

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Qwen3VLMoeConfig":
        hf = AutoConfig.from_pretrained(model_name)
        vc = hf.vision_config
        text_config = hf.get_text_config()
        rope = getattr(text_config, "rope_scaling", {}) or {}

        fp8_block_size = None
        quant_cfg = getattr(hf, "quantization_config", None)
        if quant_cfg is not None:
            quant_dict = quant_cfg if isinstance(quant_cfg, dict) else quant_cfg.to_dict()
            if quant_dict.get("quant_method") == "fp8" and "weight_block_size" in quant_dict:
                wbs = quant_dict["weight_block_size"]
                fp8_block_size = (wbs[0], wbs[1])

        return cls(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            moe_intermediate_size=text_config.moe_intermediate_size,
            num_hidden_layers=text_config.num_hidden_layers,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            head_dim=getattr(text_config, "head_dim",
                             text_config.hidden_size // text_config.num_attention_heads),
            vocab_size=text_config.vocab_size,
            max_position_embeddings=text_config.max_position_embeddings,
            rms_norm_eps=text_config.rms_norm_eps,
            rope_theta=text_config.rope_theta,
            tie_word_embeddings=hf.tie_word_embeddings,
            mrope_section=rope.get("mrope_section", [24, 20, 20]),
            mrope_interleaved=rope.get("mrope_interleaved", True),
            image_token_id=hf.image_token_id,
            video_token_id=hf.video_token_id,
            num_experts=text_config.num_experts,
            num_experts_per_tok=text_config.num_experts_per_tok,
            norm_topk_prob=getattr(text_config, "norm_topk_prob", True),
            decoder_sparse_step=getattr(text_config, "decoder_sparse_step", 1),
            vision=Qwen3VLVisionConfig(
                depth=vc.depth,
                hidden_size=vc.hidden_size,
                in_channels=vc.in_channels,
                num_heads=vc.num_heads,
                intermediate_size=vc.intermediate_size,
                hidden_act=getattr(vc, "hidden_act", "gelu_pytorch_tanh"),
                patch_size=vc.patch_size,
                spatial_merge_size=vc.spatial_merge_size,
                temporal_patch_size=vc.temporal_patch_size,
                out_hidden_size=getattr(vc, "out_hidden_size", 2048),
                deepstack_visual_indexes=getattr(vc, "deepstack_visual_indexes", [8, 16, 24]),
                num_position_embeddings=getattr(vc, "num_position_embeddings", 2304),
            ),
            fp8_block_size=fp8_block_size,
        )


class Qwen3MoeModel(nn.Module):
    def __init__(self, config: Qwen3VLMoeConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = MRotaryEmbedding(
            config.head_dim, config.max_position_embeddings,
            config.rope_theta, config.mrope_section,
            config.mrope_interleaved,
        )
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config, rotary_emb=self.rotary_emb)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, inputs_embeds=None,
                deepstack_embeds=None):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        residual = torch.zeros_like(hidden_states)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if deepstack_embeds and layer_idx < len(deepstack_embeds):
                hidden_states = hidden_states + deepstack_embeds[layer_idx]
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3VLMoeForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config: Qwen3VLMoeConfig):
        super().__init__()
        self.config = config
        self.visual = Qwen3VisionTransformer(config.vision)
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self._mrope_positions = MRopeInputPositions()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_mrope_input_positions(
        self, input_tokens: list[int],
        image_grid_thw: list[list[int]] | None = None,
        video_grid_thw: list[list[int]] | None = None,
        image_offsets: list[int] | None = None,
        video_offsets: list[int] | None = None,
    ) -> tuple[torch.Tensor, int]:
        return self._mrope_positions(
            input_tokens, self.config.vision.spatial_merge_size,
            image_grid_thw, video_grid_thw,
            image_offsets, video_offsets,
        )

    def forward(self, input_ids, positions, inputs_embeds=None,
                deepstack_embeds=None):
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds,
                          deepstack_embeds=deepstack_embeds)

    def forward_with_lm_proj(self, input_ids, positions, inputs_embeds=None):
        hidden_states = self.model(input_ids, positions, inputs_embeds=inputs_embeds)
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
        return self.lm_head.gather_greedy(partial_logits.float())
