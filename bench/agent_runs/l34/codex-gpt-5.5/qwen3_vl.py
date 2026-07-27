from __future__ import annotations

import importlib
import types
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


_BASE_CLASS = None
for _mod_name in (
    "fastkernels.tasks.baseline.L4.qwen3_vl",
    "fastkernels.tasks.baseline.qwen3_vl",
    "fastkernels.tasks.baseline.L5.qwen3_vl",
):
    try:
        _BASE_CLASS = importlib.import_module(_mod_name).Qwen3VLForConditionalGeneration
        break
    except Exception:
        pass


if triton is not None:
    @triton.jit
    def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, n_cols: tl.constexpr, eps: tl.constexpr,
                         block: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        y = x * tl.rsqrt(var + eps) * w
        tl.store(y_ptr + row * n_cols + offs, y, mask=mask)


    @triton.jit
    def _add_rms_norm_kernel(x_ptr, r_ptr, w_ptr, y_ptr, ro_ptr,
                             n_cols: tl.constexpr, eps: tl.constexpr,
                             block: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        v = x + r
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(v * v, axis=0) / n_cols
        y = v * tl.rsqrt(var + eps) * w
        tl.store(ro_ptr + row * n_cols + offs, v, mask=mask)
        tl.store(y_ptr + row * n_cols + offs, y, mask=mask)


def _num_warps(block: int) -> int:
    if block >= 4096:
        return 8
    if block >= 2048:
        return 4
    return 1


class _FastRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, residual=None):
        n_cols = x.shape[-1]
        use_triton = (
            triton is not None
            and x.is_cuda
            and self.weight.is_cuda
            and x.stride(-1) == 1
            and self.weight.is_contiguous()
            and n_cols <= 8192
        )
        if use_triton:
            x2 = x.reshape(-1, n_cols)
            y = torch.empty_like(x)
            y2 = y.reshape(-1, n_cols)
            block = triton.next_power_of_2(n_cols)
            if residual is None:
                _rms_norm_kernel[(x2.shape[0],)](
                    x2, self.weight, y2, n_cols, self.eps, block,
                    num_warps=_num_warps(block),
                )
                return y, x
            if residual.shape == x.shape and residual.stride(-1) == 1:
                r2 = residual.reshape(-1, n_cols)
                residual_out = torch.empty_like(x)
                ro2 = residual_out.reshape(-1, n_cols)
                _add_rms_norm_kernel[(x2.shape[0],)](
                    x2, r2, self.weight, y2, ro2, n_cols, self.eps, block,
                    num_warps=_num_warps(block),
                )
                return y, residual_out

        if residual is None:
            residual_out = x
            v = x
        else:
            v = x + residual
            residual_out = v
        y = v.to(torch.float32)
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = y.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)
        return y, residual_out


def _copy_fast_rms(old):
    eps = getattr(old, "eps", getattr(old, "variance_epsilon", 1e-6))
    new = _FastRMSNorm(int(old.weight.numel()), float(eps))
    new.weight = old.weight
    return new


def _replace_language_rms_norms(root):
    model = getattr(root, "model", None)
    if model is None:
        return
    norm = getattr(model, "norm", None)
    if norm is not None and hasattr(norm, "weight") and norm.weight.ndim == 1:
        model.norm = _copy_fast_rms(norm)
    for layer in getattr(model, "layers", ()):
        for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm"):
            child = getattr(layer, name, None)
            if child is not None and hasattr(child, "weight") and child.weight.ndim == 1:
                setattr(layer, name, _copy_fast_rms(child))


def _fast_model_forward(self, input_ids, positions, inputs_embeds=None, deepstack_embeds=None):
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = self.embed_tokens(input_ids)
    residual = None
    layers = self.layers
    ds_len = 0 if deepstack_embeds is None else len(deepstack_embeds)
    for layer_idx, layer in enumerate(layers):
        hidden_states, residual = layer(positions, hidden_states, residual)
        if layer_idx < ds_len:
            hidden_states = hidden_states + deepstack_embeds[layer_idx]
    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


if _BASE_CLASS is not None:
    class Qwen3VLForConditionalGeneration(_BASE_CLASS):
        def __init__(self, config, quant_config: dict | None = None):
            super().__init__(config, quant_config=quant_config)
            _replace_language_rms_norms(self)
            if hasattr(self, "model"):
                self.model.forward = types.MethodType(_fast_model_forward, self.model)

        def forward(self, input_ids, positions, inputs_embeds=None, deepstack_embeds=None):
            return _fast_model_forward(
                self.model, input_ids, positions,
                inputs_embeds=inputs_embeds, deepstack_embeds=deepstack_embeds,
            )

else:
    GELU = importlib.import_module("fastkernels.tasks.baseline.L1.gelu").GELU
    MRotaryEmbedding = importlib.import_module("fastkernels.tasks.baseline.L1.mrope").MRotaryEmbedding
    MRopeInputPositions = importlib.import_module("fastkernels.tasks.baseline.L1.mrope_input_positions").MRopeInputPositions
    SiLU = importlib.import_module("fastkernels.tasks.baseline.L1.silu").SiLU
    VisionRotaryEmbedding = importlib.import_module("fastkernels.tasks.baseline.L1.vision_rotary_emb").VisionRotaryEmbedding
    ParallelLMHead = importlib.import_module("fastkernels.tasks.baseline.L2.parallel_embedding").ParallelLMHead
    VocabParallelEmbedding = importlib.import_module("fastkernels.tasks.baseline.L2.parallel_embedding").VocabParallelEmbedding
    VisionPatchEmbed = importlib.import_module("fastkernels.tasks.baseline.L2.vision_patch_embed").VisionPatchEmbed
    VisionPatchMerger = importlib.import_module("fastkernels.tasks.baseline.L2.vision_patch_merger").VisionPatchMerger
    VisionPosEmbedInterpolate = importlib.import_module("fastkernels.tasks.baseline.L2.vision_pos_embed_interpolate").VisionPosEmbedInterpolate
    LlamaDecoderLayer = importlib.import_module("fastkernels.tasks.baseline.L3.llama_decoder").LlamaDecoderLayer
    Qwen3MoEDecoderLayer = importlib.import_module("fastkernels.tasks.baseline.L3.qwen3_moe_decoder").Qwen3MoEDecoderLayer
    VisionBlock = importlib.import_module("fastkernels.tasks.baseline.L3.vision_block").VisionBlock


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
            hidden_states = hidden_states + self.pos_embed_interp(grid_thw_list, dtype, device)
            rotary_cos, rotary_sin = self.rotary_emb(
                grid_thw_list, self.spatial_merge_size, dtype, device,
            )
            cu_seqlens = np.repeat(
                grid_thw_np[:, 1] * grid_thw_np[:, 2], grid_thw_np[:, 0]
            ).cumsum(axis=0, dtype=np.int32)
            cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
            cu_seqlens = torch.from_numpy(cu_seqlens).to(device)
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            hidden_states = hidden_states.unsqueeze(1)
            deepstack_features = []
            deepstack_map = {v: i for i, v in enumerate(self.deepstack_visual_indexes)}
            for layer_num, blk in enumerate(self.blocks):
                hidden_states = blk(hidden_states, cu_seqlens, rotary_cos, rotary_sin, max_seqlen)
                idx = deepstack_map.get(layer_num)
                if idx is not None:
                    deepstack_features.append(self.deepstack_merger_list[idx](hidden_states))
            hidden_states = self.merger(hidden_states)
            if deepstack_features:
                hidden_states = torch.cat([hidden_states] + deepstack_features, dim=1)
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
            self.norm = _FastRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            for layer in self.layers:
                for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm"):
                    child = getattr(layer, name, None)
                    if child is not None and hasattr(child, "weight") and child.weight.ndim == 1:
                        setattr(layer, name, _copy_fast_rms(child))

        def forward(self, input_ids, positions, inputs_embeds=None, deepstack_embeds=None):
            return _fast_model_forward(
                self, input_ids, positions,
                inputs_embeds=inputs_embeds, deepstack_embeds=deepstack_embeds,
            )


    class Qwen3VLForConditionalGeneration(nn.Module):
        packed_modules_mapping = {
            "q_proj": ("qkv_proj", "q"),
            "k_proj": ("qkv_proj", "k"),
            "v_proj": ("qkv_proj", "v"),
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1),
        }

        def __init__(self, config, quant_config: dict | None = None):
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
                image_grid_thw, video_grid_thw, image_offsets, video_offsets,
            )

        def forward(self, input_ids, positions, inputs_embeds=None, deepstack_embeds=None):
            return self.model(
                input_ids, positions,
                inputs_embeds=inputs_embeds, deepstack_embeds=deepstack_embeds,
            )

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
