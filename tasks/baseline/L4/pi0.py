"""Pi0 vision-language-action model (L4 pipeline).

Contains:
- SigLIPVisionEncoder: SigLIP So400m/14 vision tower.
- GemmaModel: Shared Gemma transformer (used for both VLM and action expert).
- Pi0Model: Wires VLM + DiT with KV-cache sharing.
- Pi0Pipeline: Full inference pipeline with flow-matching action generation.

L4 wiring/configuration; computation lives in L1-L3 tasks.

Reference SOTA: HuggingFace Transformers ``PI0ForConditionalGeneration``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import nn

from ..L1.conv2d import Conv2d
from ..L1.embedding import Embedding
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.rms_norm import RMSNorm
from ..L2.gemma_dense_attention import GemmaRotaryEmbedding
from ..L2.pi0_action_embed import Pi0ActionTimeEmbedding
from ..L3.gemma_dense_decoder_layer import GemmaDenseDecoderLayer
from ..L3.siglip_encoder_layer import SigLIPEncoderLayer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SigLIPVisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    image_size: int = 224
    patch_size: int = 14
    layer_norm_eps: float = 1e-6


@dataclass
class GemmaConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 18
    intermediate_size: int = 16384
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    vocab_size: int = 257152
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0


@dataclass
class Pi0Config:
    """Full Pi0 configuration."""

    vlm_text_config: GemmaConfig = field(default_factory=GemmaConfig)
    vlm_vision_config: SigLIPVisionConfig = field(default_factory=SigLIPVisionConfig)
    dit_config: GemmaConfig = field(default_factory=lambda: GemmaConfig(
        hidden_size=1024,
        intermediate_size=4096,
        head_dim=256,
    ))
    projection_dim: int = 2048
    image_token_id: int = 257152

    chunk_size: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10

    min_period: float = 0.004
    max_period: float = 4.0

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Pi0Config":
        """Load config from a HuggingFace PI0 checkpoint."""
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json at {config_path}")
        with open(config_path) as f:
            data = json.load(f)

        vlm_cfg = data.get("vlm_config", {})
        text_cfg = vlm_cfg.get("text_config", {})
        vision_cfg = vlm_cfg.get("vision_config", {})
        dit_cfg = data.get("dit_config", {})

        return cls(
            vlm_text_config=GemmaConfig(
                hidden_size=text_cfg.get("hidden_size", 2048),
                num_hidden_layers=text_cfg.get("num_hidden_layers", 18),
                intermediate_size=text_cfg.get("intermediate_size", 16384),
                num_attention_heads=text_cfg.get("num_attention_heads", 8),
                num_key_value_heads=text_cfg.get("num_key_value_heads", 1),
                head_dim=text_cfg.get("head_dim", 256),
                vocab_size=text_cfg.get("vocab_size", 257152),
                rms_norm_eps=text_cfg.get("rms_norm_eps", 1e-6),
                max_position_embeddings=text_cfg.get("max_position_embeddings", 8192),
                rope_theta=text_cfg.get("rope_theta", 10000.0),
            ),
            vlm_vision_config=SigLIPVisionConfig(
                hidden_size=vision_cfg.get("hidden_size", 1152),
                intermediate_size=vision_cfg.get("intermediate_size", 4304),
                num_hidden_layers=vision_cfg.get("num_hidden_layers", 27),
                num_attention_heads=vision_cfg.get("num_attention_heads", 16),
                image_size=vision_cfg.get("image_size", 224),
                patch_size=vision_cfg.get("patch_size", 14),
            ),
            dit_config=GemmaConfig(
                hidden_size=dit_cfg.get("hidden_size", 1024),
                num_hidden_layers=dit_cfg.get("num_hidden_layers", 18),
                intermediate_size=dit_cfg.get("intermediate_size", 4096),
                num_attention_heads=dit_cfg.get("num_attention_heads", 8),
                num_key_value_heads=dit_cfg.get("num_key_value_heads", 1),
                head_dim=dit_cfg.get("head_dim", 256),
                vocab_size=text_cfg.get("vocab_size", 257152),
                rms_norm_eps=dit_cfg.get("rms_norm_eps", 1e-6),
                max_position_embeddings=dit_cfg.get("max_position_embeddings", 8192),
                rope_theta=dit_cfg.get("rope_theta", 10000.0),
            ),
            projection_dim=vlm_cfg.get("projection_dim", 2048),
            image_token_id=vlm_cfg.get("image_token_id", 257152),
            chunk_size=data.get("chunk_size", 50),
            max_state_dim=data.get("max_state_dim", 32),
            max_action_dim=data.get("max_action_dim", 32),
            num_inference_steps=data.get("num_inference_steps", 10),
            min_period=data.get("min_period", 0.004),
            max_period=data.get("max_period", 4.0),
        )


# ---------------------------------------------------------------------------
# SigLIP Vision Encoder
# ---------------------------------------------------------------------------

class SigLIPVisionEncoder(nn.Module):
    """SigLIP So400m/14 vision tower.

    Conv2d patch embedding -> learned position embeddings -> N transformer
    encoder layers -> LayerNorm.
    """

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        num_patches = (config.image_size // config.patch_size) ** 2

        self.patch_embedding = Conv2d(
            3, config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
            bias=True,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches, config.hidden_size),
        )
        self.layers = nn.ModuleList([
            SigLIPEncoderLayer(
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.layer_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.post_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, 3, image_size, image_size)
        Returns:
            (batch, num_patches, hidden_size)
        """
        patch_embeds = self.patch_embedding(pixel_values)
        bsz, hidden, h, w = patch_embeds.shape
        patch_embeds = patch_embeds.reshape(bsz, hidden, h * w).transpose(1, 2)
        hidden_states = patch_embeds + self.position_embedding

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.post_layernorm(hidden_states)


# ---------------------------------------------------------------------------
# Gemma Transformer
# ---------------------------------------------------------------------------

class GemmaModel(nn.Module):
    """Gemma decoder-only transformer (shared by VLM and DiT).

    When used as the VLM backbone, includes token embedding.
    When used as the DiT (action expert), receives pre-embedded inputs.
    """

    def __init__(self, config: GemmaConfig, embed_tokens: bool = True):
        super().__init__()
        self.config = config
        if embed_tokens:
            self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = None
        self.layers = nn.ModuleList([
            GemmaDenseDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GemmaRotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Returns:
            hidden_states: (batch, seq, hidden_size)
            new_kv_caches: list of (key, value) per layer, or None.
        """
        if inputs_embeds is None:
            assert input_ids is not None and self.embed_tokens is not None
            hidden_states = self.embed_tokens(input_ids)
            hidden_states = hidden_states * (self.config.hidden_size ** 0.5)
        else:
            hidden_states = inputs_embeds

        bsz, seq_len = hidden_states.shape[:2]
        if position_ids is None:
            cache_len = kv_caches[0][0].shape[1] if kv_caches else 0
            position_ids = torch.arange(
                cache_len, cache_len + seq_len,
                device=hidden_states.device,
            ).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(hidden_states, position_ids)

        new_kv_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_kv = kv_caches[i] if kv_caches else None
            hidden_states, new_kv = layer(
                hidden_states, cos, sin,
                attention_mask=attention_mask, kv_cache=layer_kv,
            )
            if use_cache:
                new_kv_caches.append(new_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, new_kv_caches


# ---------------------------------------------------------------------------
# Pi0 Model (VLM + DiT with KV sharing)
# ---------------------------------------------------------------------------

class Pi0Model(nn.Module):
    """Wires VLM backbone and DiT action expert with KV-cache sharing."""

    def __init__(self, config: Pi0Config):
        super().__init__()
        self.config = config
        self.vision_tower = SigLIPVisionEncoder(config.vlm_vision_config)
        self.multi_modal_projector = Linear(
            config.vlm_vision_config.hidden_size, config.projection_dim,
        )
        self.vlm = GemmaModel(config.vlm_text_config, embed_tokens=True)
        self.dit = GemmaModel(config.dit_config, embed_tokens=False)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images through SigLIP + projector.

        Args:
            pixel_values: (batch * num_cameras, 3, H, W)
        Returns:
            (batch * num_cameras, num_patches, projection_dim)
        """
        vision_outputs = self.vision_tower(pixel_values)
        return self.multi_modal_projector(vision_outputs)

    def embed_prefix(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed image+text tokens for the VLM prefix.

        Scatters image features into the positions indicated by image_token_id.
        """
        max_num_cameras = pixel_attention_mask.shape[1]
        flat_pixels = pixel_values.flatten(0, 1)
        image_features = self.encode_images(flat_pixels)
        image_features = image_features.reshape(
            -1, max_num_cameras, image_features.shape[1], image_features.shape[2],
        )

        total_image_features = []
        for batch_idx, mask in enumerate(pixel_attention_mask):
            unpadded = image_features[batch_idx][mask]
            total_image_features.append(unpadded)
        total_image_features = torch.cat(total_image_features, dim=0)

        llm_input_ids = input_ids.clone()
        llm_input_ids[input_ids == self.config.image_token_id] = 0
        inputs_embeds = self.vlm.embed_tokens(llm_input_ids)
        inputs_embeds = inputs_embeds * (self.config.vlm_text_config.hidden_size ** 0.5)

        special_image_mask = (
            (input_ids == self.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            special_image_mask, total_image_features,
        )
        return inputs_embeds

    def forward_vlm_prefix(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Forward the VLM prefix and return cached KV.

        Returns:
            kv_caches: list of (key, value) per layer.
        """
        inputs_embeds = self.embed_prefix(input_ids, pixel_values, pixel_attention_mask)

        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(-1) - 1

        _, kv_caches = self.vlm(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=None,
            use_cache=True,
        )
        return kv_caches

    def forward_dit(
        self,
        action_embeds: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward action tokens through the DiT using VLM KV cache.

        Args:
            action_embeds: (batch, 1 + chunk_size, dit_hidden_size)
            kv_caches: VLM-prefix KV caches.
            attention_mask: Optional.

        Returns:
            (batch, 1 + chunk_size, dit_hidden_size)
        """
        prefix_len = kv_caches[0][0].shape[1] if kv_caches else 0
        suffix_len = action_embeds.shape[1]
        position_ids = torch.arange(
            prefix_len, prefix_len + suffix_len,
            device=action_embeds.device,
        ).unsqueeze(0).expand(action_embeds.shape[0], -1)

        hidden_states, _ = self.dit(
            inputs_embeds=action_embeds,
            position_ids=position_ids,
            kv_caches=kv_caches,
            use_cache=False,
        )
        return hidden_states


# ---------------------------------------------------------------------------
# Sampling outputs
# ---------------------------------------------------------------------------

@dataclass
class Pi0SamplingParams:
    """Parameters for Pi0 action generation."""
    num_inference_steps: int = 10
    seed: int | None = None


@dataclass
class Pi0Output:
    """Output of the Pi0 pipeline."""
    actions: torch.Tensor | None = None
    infer_ms: float = 0.0


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

class Pi0Pipeline(nn.Module):
    """Full Pi0 inference pipeline.

    Composes: SigLIP encode -> VLM prefix -> flow-matching action generation.
    """

    def __init__(self, config: Pi0Config):
        super().__init__()
        self.config = config
        self.model = Pi0Model(config)
        self.embed_action_time = Pi0ActionTimeEmbedding(
            expert_hidden_size=config.dit_config.hidden_size,
            max_action_dim=config.max_action_dim,
            max_state_dim=config.max_state_dim,
            min_period=config.min_period,
            max_period=config.max_period,
        )
        self.action_out_proj = Linear(
            config.dit_config.hidden_size, config.max_action_dim,
        )

    @torch.inference_mode()
    def forward(
        self,
        state: torch.Tensor,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        params: Pi0SamplingParams | None = None,
    ) -> Pi0Output:
        """Run flow-matching inference to generate actions.

        Args:
            state: (batch, max_state_dim) robot state.
            input_ids: (batch, seq_len) tokenized instruction.
            pixel_values: (batch, num_cameras, 3, H, W) camera images.
            pixel_attention_mask: (batch, num_cameras) bool mask.
            attention_mask: (batch, seq_len) text attention mask.
            noise: Optional (batch, chunk_size, max_action_dim) initial noise.
            params: Sampling parameters.

        Returns:
            Pi0Output with predicted action chunk.
        """
        import time as _time

        params = params or Pi0SamplingParams()
        num_steps = params.num_inference_steps or self.config.num_inference_steps
        batch_size = state.shape[0]
        device = state.device

        if noise is None:
            noise = torch.randn(
                batch_size, self.config.chunk_size, self.config.max_action_dim,
                dtype=pixel_values.dtype, device=device,
            )

        t0 = _time.perf_counter()

        vlm_kv = self.model.forward_vlm_prefix(
            input_ids, pixel_values, pixel_attention_mask, attention_mask,
        )
        prefix_len = vlm_kv[0][0].shape[1]

        dt = -1.0 / num_steps
        for step in range(num_steps):
            t = 1.0 + step * dt
            timestep = torch.tensor(
                t, dtype=torch.float32, device=device,
            ).expand(batch_size)

            action_embeds = self.embed_action_time(state, noise, timestep)

            dit_out = self.model.forward_dit(action_embeds, vlm_kv)

            velocity = self.action_out_proj(
                dit_out[:, -self.config.chunk_size:],
            )
            noise = noise + dt * velocity

            vlm_kv = [
                (kv[0][:, :prefix_len], kv[1][:, :prefix_len])
                for kv in vlm_kv
            ]

        elapsed_ms = (_time.perf_counter() - t0) * 1000
        return Pi0Output(actions=noise, infer_ms=elapsed_ms)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load weights from HuggingFace-format checkpoint.

        Handles name remapping between HF Transformers PI0 weight keys
        and our module hierarchy.
        """
        remap = {
            "model.vlm.vision_tower.vision_model.embeddings.patch_embedding.": "model.vision_tower.patch_embedding.",
            "model.vlm.vision_tower.vision_model.embeddings.position_embedding.": "_position_embedding_",
            "model.vlm.vision_tower.vision_model.encoder.layers.": "model.vision_tower.layers.",
            "model.vlm.vision_tower.vision_model.post_layernorm.": "model.vision_tower.post_layernorm.",
            "model.vlm.multi_modal_projector.linear.": "model.multi_modal_projector.",
            "model.vlm.language_model.model.embed_tokens.": "model.vlm.embed_tokens.emb.",
            "model.vlm.language_model.model.layers.": "model.vlm.layers.",
            "model.vlm.language_model.model.norm.": "model.vlm.norm.",
            "model.dit.model.layers.": "model.dit.layers.",
            "model.dit.model.norm.": "model.dit.norm.",
            "embed_action_time.sinusoid_embeds.": "embed_action_time.sinusoid_embeds.",
            "embed_action_time.action_in_proj.": "embed_action_time.action_in_proj.",
            "embed_action_time.state_proj.": "embed_action_time.state_proj.",
            "embed_action_time.action_time_mlp_in.": "embed_action_time.action_time_mlp_in.",
            "embed_action_time.action_time_mlp_out.": "embed_action_time.action_time_mlp_out.",
            "action_out_proj.": "action_out_proj.",
        }

        siglip_attn_remap = {
            ".self_attn.q_proj.": ".self_attn.q_proj.",
            ".self_attn.k_proj.": ".self_attn.k_proj.",
            ".self_attn.v_proj.": ".self_attn.v_proj.",
            ".self_attn.out_proj.": ".self_attn.out_proj.",
        }

        gemma_layer_remap = {
            ".self_attn.q_proj.": ".self_attn.q_proj.",
            ".self_attn.k_proj.": ".self_attn.k_proj.",
            ".self_attn.v_proj.": ".self_attn.v_proj.",
            ".self_attn.o_proj.": ".self_attn.o_proj.",
            ".mlp.gate_proj.": ".mlp.gate_up_proj.",
            ".mlp.up_proj.": ".mlp.gate_up_proj.",
            ".mlp.down_proj.": ".mlp.down_proj.",
        }

        params_dict = dict(self.named_parameters())
        for name, buf in self.named_buffers():
            params_dict[name] = buf

        loaded = set()

        for name, tensor in weights:
            original_name = name

            if name == "model.vlm.vision_tower.vision_model.embeddings.position_embedding.weight":
                self.model.vision_tower.position_embedding.data.copy_(
                    tensor.unsqueeze(0) if tensor.ndim == 2 else tensor,
                )
                loaded.add(original_name)
                continue

            mapped = name
            for src, dst in remap.items():
                if name.startswith(src):
                    mapped = name.replace(src, dst, 1)
                    break

            is_gate = ".mlp.gate_proj." in name
            is_up = ".mlp.up_proj." in name

            if is_gate or is_up:
                gate_up_name = mapped.replace(".mlp.gate_proj.", ".mlp.gate_up_proj.").replace(
                    ".mlp.up_proj.", ".mlp.gate_up_proj.",
                )
                if gate_up_name in params_dict:
                    param = params_dict[gate_up_name]
                    mid = param.shape[0] // 2
                    if is_gate:
                        param.data[:mid].copy_(tensor)
                    else:
                        param.data[mid:].copy_(tensor)
                    loaded.add(original_name)
                    continue

            if mapped in params_dict:
                param = params_dict[mapped]
                if hasattr(param, "weight_loader"):
                    param.weight_loader(param, tensor)
                else:
                    param.data.copy_(tensor)
                loaded.add(original_name)
            elif "lm_head" in name or "language_model.lm_head" in name:
                loaded.add(original_name)
            else:
                logger.debug("Skipping unrecognized weight: %s", name)

        return loaded
