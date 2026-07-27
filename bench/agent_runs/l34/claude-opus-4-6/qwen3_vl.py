from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig

from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.mrope import MRotaryEmbedding
from fastkernels.tasks.baseline.L1.mrope_input_positions import MRopeInputPositions
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.vision_rotary_emb import VisionRotaryEmbedding
from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from fastkernels.tasks.baseline.L2.vision_patch_embed import VisionPatchEmbed
from fastkernels.tasks.baseline.L2.vision_patch_merger import VisionPatchMerger
from fastkernels.tasks.baseline.L2.vision_pos_embed_interpolate import VisionPosEmbedInterpolate
from fastkernels.tasks.baseline.L3.llama_decoder import LlamaDecoderLayer
from fastkernels.tasks.baseline.L3.qwen3_moe_decoder import Qwen3MoEDecoderLayer
from fastkernels.tasks.baseline.L3.vision_block import VisionBlock


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
    num_hidden_layers: int = 94
    num_attention_heads: int = 64
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
    vision: Qwen3VLVisionConfig = field(default_factory=Qwen3VLVisionConfig)
    dtype: torch.dtype = torch.bfloat16
    is_moe: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Qwen3VLConfig":
        hf = AutoConfig.from_pretrained(model_name)
        vc = hf.vision_config
        text_config = hf.get_text_config()
        rope = getattr(text_config, "rope_scaling", None) or getattr(text_config, "rope_parameters", None) or {}
        rope_theta = getattr(text_config, "rope_theta", None) or rope.get("rope_theta", 5000000.0)
        is_moe = hasattr(text_config, "num_experts") and text_config.num_experts > 0
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
            rope_theta=rope_theta,
            tie_word_embeddings=text_config.tie_word_embeddings,
            mrope_section=rope.get("mrope_section", [24, 20, 20]),
            mrope_interleaved=rope.get("mrope_interleaved", True),
            image_token_id=hf.image_token_id,
            video_token_id=hf.video_token_id,
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
            is_moe=is_moe,
            num_experts=getattr(text_config, "num_experts", 0) if is_moe else 0,
            num_experts_per_tok=getattr(text_config, "num_experts_per_tok", 0) if is_moe else 0,
            moe_intermediate_size=getattr(text_config, "moe_intermediate_size", 0) if is_moe else 0,
            norm_topk_prob=getattr(text_config, "norm_topk_prob", True) if is_moe else True,
        )


_ACTIVATION_MAP = {
    "silu": SiLU(),
    "gelu": GELU(),
    "gelu_pytorch_tanh": GELU(approximate="tanh"),
}


class Qwen3VisionTransformer(nn.Module):
    def __init__(self, vision_config: Qwen3VLVisionConfig):
        super().__init__()
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self._deepstack_index_map = {
            layer_num: idx
            for idx, layer_num in enumerate(vision_config.deepstack_visual_indexes)
        }
        self.out_hidden_size = vision_config.out_hidden_size * (
            1 + len(self.deepstack_visual_indexes)
        )
        self.patch_embed = VisionPatchEmbed(
            vision_config.patch_size, vision_config.temporal_patch_size,
            vision_config.in_channels, vision_config.hidden_size, bias=True,
        )
        self.pos_embed_interp = VisionPosEmbedInterpolate(
            vision_config.num_position_embeddings,
            vision_config.hidden_size,
            vision_config.spatial_merge_size,
        )
        head_dim = vision_config.hidden_size // vision_config.num_heads
        self.rotary_emb = VisionRotaryEmbedding(head_dim // 2)
        act_fn = _ACTIVATION_MAP.get(vision_config.hidden_act, SiLU())
        self.blocks = nn.ModuleList([
            VisionBlock(
                vision_config.hidden_size, vision_config.num_heads,
                vision_config.intermediate_size, act_fn=act_fn,
            )
            for _ in range(vision_config.depth)
        ])
        self.merger = VisionPatchMerger(
            vision_config.out_hidden_size, vision_config.hidden_size,
            vision_config.spatial_merge_size,
        )
        self.deepstack_merger_list = nn.ModuleList([
            VisionPatchMerger(
                vision_config.out_hidden_size, vision_config.hidden_size,
                vision_config.spatial_merge_size, use_postshuffle_norm=True,
            )
            for _ in range(len(self.deepstack_visual_indexes))
        ])

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

        pos_embeds = self.pos_embed_interp(grid_thw_list, dtype, device)
        hidden_states.add_(pos_embeds)

        rotary_cos, rotary_sin = self.rotary_emb(
            grid_thw_list, self.spatial_merge_size, dtype, device,
        )

        seq_lens = np.repeat(
            grid_thw_np[:, 1] * grid_thw_np[:, 2], grid_thw_np[:, 0]
        )
        max_seqlen = int(seq_lens.max())
        cu_seqlens_np = np.empty(len(seq_lens) + 1, dtype=np.int32)
        cu_seqlens_np[0] = 0
        np.cumsum(seq_lens, axis=0, dtype=np.int32, out=cu_seqlens_np[1:])
        cu_seqlens = torch.from_numpy(cu_seqlens_np).to(device)

        hidden_states = hidden_states.unsqueeze(1)
        deepstack_index_map = self._deepstack_index_map
        num_deepstack = len(deepstack_index_map)
        deepstack_merger_list = self.deepstack_merger_list

        if num_deepstack > 0 and x.is_cuda:
            deepstack_features = [None] * num_deepstack
            side_stream = torch.cuda.Stream(device=device)
            current_stream = torch.cuda.current_stream(device)
            for layer_num, blk in enumerate(self.blocks):
                hidden_states = blk(hidden_states, cu_seqlens,
                                    rotary_cos, rotary_sin, max_seqlen)
                idx = deepstack_index_map.get(layer_num)
                if idx is not None:
                    hs_ref = hidden_states
                    merger = deepstack_merger_list[idx]
                    event = torch.cuda.Event()
                    event.record(current_stream)
                    with torch.cuda.stream(side_stream):
                        side_stream.wait_event(event)
                        deepstack_features[idx] = merger(hs_ref)
            hidden_states = self.merger(hidden_states)
            current_stream.wait_stream(side_stream)
            hidden_states = torch.cat(
                [hidden_states] + deepstack_features, dim=1
            )
        elif num_deepstack > 0:
            deepstack_features = [None] * num_deepstack
            for layer_num, blk in enumerate(self.blocks):
                hidden_states = blk(hidden_states, cu_seqlens,
                                    rotary_cos, rotary_sin, max_seqlen)
                idx = deepstack_index_map.get(layer_num)
                if idx is not None:
                    deepstack_features[idx] = deepstack_merger_list[idx](hidden_states)
            hidden_states = self.merger(hidden_states)
            hidden_states = torch.cat(
                [hidden_states] + deepstack_features, dim=1
            )
        else:
            for blk in self.blocks:
                hidden_states = blk(hidden_states, cu_seqlens,
                                    rotary_cos, rotary_sin, max_seqlen)
            hidden_states = self.merger(hidden_states)
        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3VLConfig, quant_config: dict | None = None):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = MRotaryEmbedding(
            config.head_dim, config.max_position_embeddings,
            config.rope_theta, config.mrope_section,
            config.mrope_interleaved,
        )
        if config.is_moe:
            self.layers = nn.ModuleList([
                Qwen3MoEDecoderLayer(config, rotary_emb=self.rotary_emb,
                                     quant_config=quant_config)
                for _ in range(config.num_hidden_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                LlamaDecoderLayer(config, rotary_emb=self.rotary_emb, qk_norm=True,
                                  quant_config=quant_config)
                for _ in range(config.num_hidden_layers)
            ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, inputs_embeds=None,
                deepstack_embeds=None):
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        residual = None
        layers = self.layers
        if deepstack_embeds:
            ds_len = len(deepstack_embeds)
            num_layers = len(layers)
            ds_end = min(ds_len, num_layers)
            for i in range(ds_end):
                hidden_states, residual = layers[i](positions, hidden_states, residual)
                hidden_states = hidden_states + deepstack_embeds[i]
            for i in range(ds_end, num_layers):
                hidden_states, residual = layers[i](positions, hidden_states, residual)
        else:
            for layer in layers:
                hidden_states, residual = layer(positions, hidden_states, residual)
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

    def __init__(self, config: Qwen3VLConfig, quant_config: dict | None = None):
        if config.is_moe:
            self.packed_modules_mapping = {
                "q_proj": ("qkv_proj", "q"),
                "k_proj": ("qkv_proj", "k"),
                "v_proj": ("qkv_proj", "v"),
            }
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.visual = Qwen3VisionTransformer(config.vision)
        self.model = Qwen3Model(config, quant_config=quant_config)
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
        video_second_per_grid: list[float] | None = None,
        audio_feature_lengths=None,
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
