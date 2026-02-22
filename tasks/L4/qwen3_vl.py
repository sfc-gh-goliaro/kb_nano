"""Qwen3-VL model: vision encoder with DeepStack + Qwen3 language model with M-RoPE.

Supports image and video inputs through the vision encoder pipeline.
Key differences from Qwen2-VL:
- Vision encoder uses SiLU activation, learned position embeddings, DeepStack
- Language model uses QK-norm (per-head RMSNorm) instead of QKV bias
- mrope_interleaved=True for Qwen3-VL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from ..L1.mrope import MRotaryEmbedding
from ..L1.rms_norm import RMSNorm
from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L2.parallel_linear import ColumnParallelLinear, RowParallelLinear
from ..L2.vision_attention import VisionAttention
from ..L2.vision_mlp import Qwen3VisionMLP
from ..L3.qwen3_decoder import Qwen3DecoderLayer


@dataclass
class Qwen3VLVisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    in_channels: int = 3
    num_heads: int = 16
    intermediate_size: int = 4304
    hidden_act: str = "gelu_pytorch_tanh"
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 4096
    deepstack_visual_indexes: list[int] = field(default_factory=lambda: [8, 16, 24])
    num_position_embeddings: int = 2304


@dataclass
class Qwen3VLConfig:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    tie_word_embeddings: bool = False
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    mrope_interleaved: bool = True
    vision: Qwen3VLVisionConfig = field(default_factory=Qwen3VLVisionConfig)
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Qwen3VLConfig":
        hf = AutoConfig.from_pretrained(model_name)
        vc = hf.vision_config
        text_config = hf.get_text_config()
        rope = getattr(text_config, "rope_scaling", {}) or {}
        return cls(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            num_hidden_layers=text_config.num_hidden_layers,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            head_dim=getattr(text_config, "head_dim",
                             text_config.hidden_size // text_config.num_attention_heads),
            vocab_size=text_config.vocab_size,
            max_position_embeddings=text_config.max_position_embeddings,
            rms_norm_eps=text_config.rms_norm_eps,
            rope_theta=text_config.rope_theta,
            tie_word_embeddings=text_config.tie_word_embeddings,
            mrope_section=rope.get("mrope_section", [24, 20, 20]),
            mrope_interleaved=rope.get("mrope_interleaved", True),
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
                out_hidden_size=getattr(vc, "out_hidden_size", 4096),
                deepstack_visual_indexes=getattr(vc, "deepstack_visual_indexes", [8, 16, 24]),
                num_position_embeddings=getattr(vc, "num_position_embeddings", 2304),
            ),
        )


# ---- Vision Encoder Components ----

_ACTIVATION_MAP = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
}


class Qwen3VisionPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, temporal_patch_size: int,
                 in_channels: int, hidden_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        self.input_size = in_channels * temporal_patch_size * patch_size * patch_size
        kernel = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, hidden_size, kernel, stride=kernel, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, _ = x.shape
        x = x.view(L, self.input_size)
        return F.linear(
            x,
            self.proj.weight.view(self.hidden_size, self.input_size),
            self.proj.bias,
        )


class Qwen3VisionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden_dim: int,
                 act_fn=F.silu, norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = VisionAttention(dim, num_heads)
        self.mlp = Qwen3VisionMLP(dim, mlp_hidden_dim, act_fn=act_fn)

    def forward(self, x, cu_seqlens, rotary_pos_emb_cos,
                rotary_pos_emb_sin, max_seqlen=None):
        x = x + self.attn(self.norm1(x), cu_seqlens,
                          rotary_pos_emb_cos, rotary_pos_emb_sin, max_seqlen)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VisionPatchMerger(nn.Module):
    def __init__(self, d_model: int, context_dim: int,
                 spatial_merge_size: int = 2,
                 use_postshuffle_norm: bool = False):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.mlp = nn.ModuleList([
            ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True),
            nn.GELU(),
            RowParallelLinear(self.hidden_size, d_model, bias=True),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        fc1, act, fc2 = self.mlp
        x = fc2(act(fc1(x)))
        return x


class Qwen3VisionTransformer(nn.Module):
    def __init__(self, vision_config: Qwen3VLVisionConfig):
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.out_hidden_size = vision_config.out_hidden_size * (
            1 + len(self.deepstack_visual_indexes)
        )

        self.patch_embed = Qwen3VisionPatchEmbed(
            vision_config.patch_size, vision_config.temporal_patch_size,
            vision_config.in_channels, vision_config.hidden_size,
        )

        self.pos_embed = nn.Embedding(
            vision_config.num_position_embeddings, vision_config.hidden_size,
        )
        self.num_grid_per_side = int(vision_config.num_position_embeddings ** 0.5)

        head_dim = vision_config.hidden_size // vision_config.num_heads
        self.rotary_dim = head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
        ))
        t = torch.arange(8192, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        act_fn = _ACTIVATION_MAP.get(vision_config.hidden_act, F.silu)

        self.blocks = nn.ModuleList([
            Qwen3VisionBlock(
                vision_config.hidden_size, vision_config.num_heads,
                vision_config.intermediate_size, act_fn=act_fn,
            )
            for _ in range(vision_config.depth)
        ])

        self.merger = Qwen3VisionPatchMerger(
            vision_config.out_hidden_size, vision_config.hidden_size,
            vision_config.spatial_merge_size,
        )
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VisionPatchMerger(
                vision_config.out_hidden_size, vision_config.hidden_size,
                vision_config.spatial_merge_size, use_postshuffle_norm=True,
            )
            for _ in range(len(self.deepstack_visual_indexes))
        ])

    def rot_pos_emb(self, grid_thw_list):
        sms = self.spatial_merge_size
        pos_ids = []
        max_grid_size = 0
        for t, h, w in grid_thw_list:
            hpos = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
            wpos = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
            hpos = hpos.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).flatten()
            wpos = wpos.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).flatten()
            hw = np.stack([hpos, wpos], axis=-1)
            pos_ids.append(np.tile(hw, (t, 1)) if t > 1 else hw)
            max_grid_size = max(max_grid_size, h, w)
        pos_ids = torch.from_numpy(np.concatenate(pos_ids, axis=0)).to(
            self.patch_embed.proj.weight.device
        )

        cache = self.cos_sin_cache[:max_grid_size].to(
            dtype=self.patch_embed.proj.weight.dtype
        )
        cos, sin = cache.chunk(2, dim=-1)
        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)
        return cos_combined, sin_combined

    def fast_pos_embed_interpolate(self, grid_thw_list):
        """Bilinear interpolation of learned position embeddings."""
        num_grid = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim
        device = self.patch_embed.proj.weight.device
        dtype = self.patch_embed.proj.weight.dtype

        outputs = []
        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, num_grid - 1, h, dtype=torch.float32, device=device)
            w_idxs = torch.linspace(0, num_grid - 1, w, dtype=torch.float32, device=device)

            h_floor = h_idxs.long()
            w_floor = w_idxs.long()
            h_ceil = torch.clamp(h_floor + 1, max=num_grid - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            indices = (h_grid * num_grid + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1).to(dtype=dtype)

            embeds = self.pos_embed(indices) * weights
            combined = embeds.sum(dim=0)
            combined = combined.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            ).permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor | list):
        device = self.patch_embed.proj.weight.device
        dtype = self.patch_embed.proj.weight.dtype
        x = x.to(device=device, dtype=dtype)
        hidden_states = self.patch_embed(x)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw_np = np.array(grid_thw, dtype=np.int32)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_thw_np = grid_thw.numpy()

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw_list)
        hidden_states = hidden_states + pos_embeds

        rotary_cos, rotary_sin = self.rot_pos_emb(grid_thw_list)

        cu_seqlens = np.repeat(
            grid_thw_np[:, 1] * grid_thw_np[:, 2], grid_thw_np[:, 0]
        ).cumsum(axis=0, dtype=np.int32)
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
        cu_seqlens = torch.from_numpy(cu_seqlens).to(device)
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        hidden_states = hidden_states.unsqueeze(1)
        deepstack_features = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens,
                                rotary_cos, rotary_sin, max_seqlen)
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_features.append(
                    self.deepstack_merger_list[idx](hidden_states)
                )

        hidden_states = self.merger(hidden_states)
        if deepstack_features:
            return hidden_states, deepstack_features
        return hidden_states, []


# ---- Language Model ----

class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MRotaryEmbedding(
            config.head_dim, config.max_position_embeddings,
            config.rope_theta, config.mrope_section,
            config.mrope_interleaved,
        )

    def forward(self, input_ids, positions, inputs_embeds=None,
                deepstack_embeds=None):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions, hidden_states, residual, self.rotary_emb,
            )
            if deepstack_embeds and layer_idx < len(deepstack_embeds):
                hidden_states = hidden_states + deepstack_embeds[layer_idx]
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3VLForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.visual = Qwen3VisionTransformer(config.vision)
        self.model = Qwen3Model(config)
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
