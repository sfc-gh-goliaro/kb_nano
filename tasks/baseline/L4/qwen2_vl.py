"""Qwen2-VL model: vision encoder + Qwen2 language model with M-RoPE.

Supports image and video inputs through the vision encoder pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L1.mrope import MRotaryEmbedding
from ..L1.rms_norm import RMSNorm
from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L2.vision_patch_embed import VisionPatchEmbed
from ..L2.vision_patch_merger import VisionPatchMerger
from ..L3.qwen2_decoder import Qwen2DecoderLayer
from ..L3.vision_block import VisionBlock


@dataclass
class Qwen2VLVisionConfig:
    depth: int = 32
    embed_dim: int = 1280
    hidden_size: int = 3584
    in_channels: int = 3
    num_heads: int = 16
    mlp_ratio: float = 4.0
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2


@dataclass
class Qwen2VLConfig:
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = False
    mrope_section: list[int] = field(default_factory=lambda: [16, 24, 24])
    mrope_interleaved: bool = False
    vision: Qwen2VLVisionConfig = field(default_factory=Qwen2VLVisionConfig)
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Qwen2VLConfig":
        hf = AutoConfig.from_pretrained(model_name)
        vc = hf.vision_config
        rope = getattr(hf, "rope_scaling", {}) or {}
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
            rope_theta=hf.rope_theta,
            tie_word_embeddings=hf.tie_word_embeddings,
            mrope_section=rope.get("mrope_section", [16, 24, 24]),
            mrope_interleaved=rope.get("mrope_interleaved", False),
            vision=Qwen2VLVisionConfig(
                depth=vc.depth,
                embed_dim=vc.embed_dim,
                hidden_size=vc.hidden_size,
                in_channels=vc.in_channels,
                num_heads=vc.num_heads,
                mlp_ratio=vc.mlp_ratio,
                patch_size=vc.patch_size,
                spatial_merge_size=vc.spatial_merge_size,
                temporal_patch_size=vc.temporal_patch_size,
            ),
        )


# ---- Vision Encoder Components ----

class Qwen2VisionTransformer(nn.Module):
    def __init__(self, vision_config: Qwen2VLVisionConfig):
        super().__init__()
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.num_heads = vision_config.num_heads
        self.embed_dim = vision_config.embed_dim

        self.patch_embed = VisionPatchEmbed(
            vision_config.patch_size, vision_config.temporal_patch_size,
            vision_config.in_channels, vision_config.embed_dim,
        )

        head_dim = vision_config.embed_dim // vision_config.num_heads
        self.rotary_dim = head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        t = torch.arange(8192, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        self.blocks = nn.ModuleList([
            VisionBlock(vision_config.embed_dim, vision_config.num_heads,
                        int(vision_config.embed_dim * vision_config.mlp_ratio))
            for _ in range(vision_config.depth)
        ])
        self.merger = VisionPatchMerger(
            vision_config.hidden_size, vision_config.embed_dim,
            vision_config.spatial_merge_size,
        )

    def rot_pos_emb(self, grid_thw_list):
        pos_ids = []
        max_grid_size = 0
        sms = self.spatial_merge_size
        for t, h, w in grid_thw_list:
            hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos = hpos.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).flatten()
            wpos = wpos.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
            max_grid_size = max(max_grid_size, h, w)
        pos_ids = torch.cat(pos_ids, dim=0)

        cache = self.cos_sin_cache[:max_grid_size].to(dtype=self.patch_embed.proj.weight.dtype)
        cos, sin = cache.chunk(2, dim=-1)
        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)
        return cos_combined, sin_combined

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor | list):
        x = x.to(device=self.patch_embed.proj.weight.device,
                  dtype=self.patch_embed.proj.weight.dtype)
        x = self.patch_embed(x)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw_np = np.array(grid_thw, dtype=np.int32)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_thw_np = grid_thw.numpy()

        rotary_cos, rotary_sin = self.rot_pos_emb(grid_thw_list)

        cu_seqlens = np.repeat(
            grid_thw_np[:, 1] * grid_thw_np[:, 2], grid_thw_np[:, 0]
        ).cumsum(axis=0, dtype=np.int32)
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
        cu_seqlens = torch.from_numpy(cu_seqlens).to(x.device)
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        x = x.unsqueeze(1)  # (seq, 1, dim) for batch dimension
        for blk in self.blocks:
            x = blk(x, cu_seqlens, rotary_cos, rotary_sin, max_seqlen)

        x = self.merger(x)
        return x


# ---- Language Model ----

class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MRotaryEmbedding(
            config.head_dim, config.max_position_embeddings,
            config.rope_theta, config.mrope_section,
            config.mrope_interleaved,
        )

    def forward(self, input_ids, positions, inputs_embeds=None):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual, self.rotary_emb,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2VLForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen2VLConfig):
        super().__init__()
        self.config = config
        self.visual = Qwen2VisionTransformer(config.vision)
        self.model = Qwen2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_mrope_input_positions(
        self, input_tokens: list[int],
        image_grid_thw: list[list[int]] | None = None,
        video_grid_thw: list[list[int]] | None = None,
        image_offsets: list[int] | None = None,
        video_offsets: list[int] | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Compute M-RoPE 3D positions for text+vision tokens."""
        spatial_merge_size = self.config.vision.spatial_merge_size
        llm_pos_ids_list = []
        st = 0

        media_items = []
        if image_grid_thw and image_offsets:
            for i, (t, h, w) in enumerate(image_grid_thw):
                merged_h = h // spatial_merge_size
                merged_w = w // spatial_merge_size
                media_items.append((image_offsets[i], t, merged_h, merged_w))
        if video_grid_thw and video_offsets:
            for i, (t, h, w) in enumerate(video_grid_thw):
                merged_h = h // spatial_merge_size
                merged_w = w // spatial_merge_size
                media_items.append((video_offsets[i], t, merged_h, merged_w))
        media_items.sort(key=lambda x: x[0])

        for offset, grid_t, grid_h, grid_w in media_items:
            text_len = offset - st
            st_idx = int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )
            grid_indices = np.indices((grid_t, grid_h, grid_w))
            llm_pos_ids_list.append(grid_indices.reshape(3, -1) + text_len + st_idx)
            st = offset + grid_t * grid_h * grid_w

        if st < len(input_tokens):
            st_idx = int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

        if not llm_pos_ids_list:
            positions = np.broadcast_to(
                np.arange(len(input_tokens)), (3, len(input_tokens))
            )
            return torch.from_numpy(positions), 0

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mrope_position_delta = int(llm_positions.max() + 1 - len(input_tokens))
        return torch.from_numpy(llm_positions), mrope_position_delta

    def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

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
