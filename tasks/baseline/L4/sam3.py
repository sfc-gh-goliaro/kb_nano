"""SAM3 image segmentation pipeline (L4 pipeline).

Contains:
- Sam3Config: model configuration dataclass.
- Sam3Model: full SAM3 image segmentation model wiring VL backbone, fusion
  encoder, detection decoder, geometry encoder, and segmentation head.

L4 wiring/configuration; computation lives in L1-L3 tasks.

Reference: sam3/model/sam3_image.py Sam3Image
           sam3/model_builder.py build_sam3_image_model
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.sam3_position_encoding import Sam3PositionEncoding
from ..L2.sam3_cross_attention import Sam3CrossAttention
from ..L2.sam3_mask_predictor import Sam3MaskPredictor
from ..L3.sam3_decoder_layer import Sam3DecoderLayer
from ..L3.sam3_encoder_layer import Sam3EncoderLayer
from ..L3.sam3_neck import Sam3Neck
from ..L3.sam3_pixel_decoder import Sam3PixelDecoder
from ..L3.sam3_text_encoder import Sam3TextEncoder
from ..L3.sam3_vit import Sam3ViT


@dataclass
class Sam3Config:
    """Configuration for SAM3 image segmentation model."""

    # ViT backbone
    img_size: int = 1008
    patch_size: int = 14
    embed_dim: int = 1024
    vit_depth: int = 32
    vit_num_heads: int = 16
    vit_mlp_ratio: float = 4.625
    window_size: int = 24
    global_att_blocks: tuple = (7, 15, 23, 31)
    use_rope: bool = True
    use_tiled_rope: bool = True
    pretrain_img_size: int = 336
    retain_cls_token: bool = False
    ln_pre: bool = True
    bias_patch_embed: bool = False

    # Neck / FPN
    d_model: int = 256
    scale_factors: tuple = (4.0, 2.0, 1.0, 0.5)
    scalp: int = 1

    # Text encoder
    text_width: int = 1024
    text_heads: int = 16
    text_layers: int = 24
    text_context_length: int = 32
    text_vocab_size: int = 49408

    # Fusion encoder
    encoder_layers: int = 6
    encoder_dim_feedforward: int = 2048
    encoder_n_head: int = 8

    # Detection decoder
    decoder_layers: int = 6
    decoder_dim_feedforward: int = 2048
    decoder_n_head: int = 8
    num_queries: int = 200
    text_cross_attention: bool = True

    # Segmentation head
    pixel_decoder_upsampling_stages: int = 3
    presence_head: bool = True

    # Box head
    box_head_hidden: int = 256

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Sam3Config":
        """Create default config for known SAM3 models."""
        return cls()


class Sam3FusionEncoder(nn.Module):
    """Multi-layer fusion encoder that fuses text and image features.

    Uses only the last feature level (matching reference num_feature_levels=1)
    and does NOT add pooled text to image features (matching reference
    add_pooled_text_to_img_feat=False).
    """

    def __init__(self, config: Sam3Config):
        super().__init__()
        self.num_feature_levels = 1
        self.layers = nn.ModuleList([
            Sam3EncoderLayer(
                config.d_model,
                config.encoder_n_head,
                config.encoder_dim_feedforward,
            )
            for _ in range(config.encoder_layers)
        ])
        self.text_pooling_proj = Linear(config.d_model, config.d_model, bias=True)
        self.level_embed = None

    def forward(
        self,
        src: List[torch.Tensor],
        src_pos: List[torch.Tensor],
        prompt: torch.Tensor,
        prompt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run fusion encoder.

        Args:
            src: List of (B, C, H, W) image feature maps (last level used).
            src_pos: Corresponding position encodings.
            prompt: (seq, B, D) text features.
            prompt_key_padding_mask: (B, seq) padding mask.

        Returns:
            Dict with 'memory', 'memory_text', 'spatial_shapes', etc.
        """
        feat = src[-1]
        pos = src_pos[-1]

        b, c, h, w = feat.shape
        spatial_shapes = [(h, w)]
        src_flat = feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        pos_flat = pos.flatten(2).transpose(1, 2)

        prompt_bf = prompt.transpose(0, 1)  # (B, seq, D)

        output = src_flat
        for layer in self.layers:
            output = layer(
                tgt=output,
                memory=prompt_bf,
                query_pos=pos_flat,
                memory_key_padding_mask=prompt_key_padding_mask,
            )

        return {
            "memory": output.transpose(0, 1),  # (HW, B, D)
            "memory_text": prompt,
            "pos_embed": pos_flat.transpose(0, 1),
            "spatial_shapes": spatial_shapes,
        }


class Sam3Decoder(nn.Module):
    """Multi-layer detection decoder with learnable queries and box heads.

    Matches reference: learned reference_points, shared bbox_embed MLP,
    query_pos only used inside attention (via with_pos_embed), and
    memory_pos passed to decoder layers for image cross-attention keys.
    """

    def __init__(self, config: Sam3Config):
        super().__init__()
        self.d_model = config.d_model
        self.num_queries = config.num_queries
        self.num_layers = config.decoder_layers

        self.query_embed = nn.Embedding(config.num_queries, config.d_model)
        self.reference_points = nn.Embedding(config.num_queries, 4)

        self.layers = nn.ModuleList([
            Sam3DecoderLayer(
                config.d_model,
                config.decoder_n_head,
                config.decoder_dim_feedforward,
                text_cross_attention=config.text_cross_attention,
            )
            for _ in range(config.decoder_layers)
        ])
        self.norm = LayerNorm(config.d_model)

        self.bbox_embed = nn.Sequential(
            Linear(config.d_model, config.d_model, bias=True),
            nn.ReLU(),
            Linear(config.d_model, config.d_model, bias=True),
            nn.ReLU(),
            Linear(config.d_model, 4, bias=True),
        )

        self.ref_point_head = nn.Sequential(
            Linear(2 * config.d_model, config.d_model, bias=True),
            nn.ReLU(),
            Linear(config.d_model, config.d_model, bias=True),
        )

    def forward(
        self,
        memory: torch.Tensor,
        pos_embed: torch.Tensor,
        memory_text: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run detection decoder.

        Args:
            memory: (L, B, D) encoder memory.
            pos_embed: (L, B, D) position embeddings.
            memory_text: (S, B, D) text features.
            text_attention_mask: (B, S) padding mask.

        Returns:
            (hs, pred_boxes):
                hs: (B, Q, D) final decoder hidden states.
                pred_boxes: (B, Q, 4) predicted boxes (sigmoid).
        """
        B = memory.shape[1]
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        memory_bf = memory.transpose(0, 1)  # (B, L, D)
        memory_pos_bf = pos_embed.transpose(0, 1)  # (B, L, D)
        text_bf = memory_text.transpose(0, 1) if memory_text is not None else None

        reference_boxes = self.reference_points.weight.unsqueeze(0).expand(B, -1, -1).sigmoid()

        intermediate_hs = []
        intermediate_ref_boxes = [reference_boxes]

        for layer in self.layers:
            query_sine_embed = self._gen_sineembed(reference_boxes)  # (B, Q, 2*d_model)
            query_pos = self.ref_point_head(query_sine_embed)  # (B, Q, d_model)

            tgt = layer(
                tgt=tgt,
                memory=memory_bf,
                tgt_query_pos=query_pos,
                memory_pos=memory_pos_bf,
                memory_text=text_bf,
                text_attention_mask=text_attention_mask,
            )

            delta_unsig = self.bbox_embed(self.norm(tgt))
            new_ref = (self._inverse_sigmoid(reference_boxes) + delta_unsig).sigmoid()
            reference_boxes = new_ref.detach()
            intermediate_ref_boxes.append(new_ref)
            intermediate_hs.append(self.norm(tgt))

        hs = intermediate_hs[-1]  # (B, Q, D)
        return hs, reference_boxes

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    @staticmethod
    def _gen_sineembed(pos: torch.Tensor, num_feats: int = 256) -> torch.Tensor:
        """Generate sine positional embedding from normalized coordinates.

        Matches reference gen_sineembed_for_position exactly.

        Args:
            pos: (..., 4) normalized cxcywh in [0, 1].
            num_feats: Total feature dim (halved internally per coord).

        Returns:
            (..., 4 * num_feats // 2) = (..., 2*d_model) sine embedding.
        """
        half = num_feats // 2
        scale = 2 * math.pi
        dim_t = torch.arange(half, dtype=torch.float32, device=pos.device)
        dim_t = 10000.0 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / half)

        x_embed = pos[..., 0] * scale
        y_embed = pos[..., 1] * scale
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

        w_embed = pos[..., 2] * scale
        h_embed = pos[..., 3] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_h = h_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(-2)

        return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1).to(pos.dtype)


class Sam3SegmentationHead(nn.Module):
    """Segmentation head combining pixel decoder, mask predictor,
    and prompt cross-attention (matching reference UniversalSegmentationHead)."""

    def __init__(self, config: Sam3Config):
        super().__init__()
        self.d_model = config.d_model
        self.pixel_decoder = Sam3PixelDecoder(
            config.d_model, config.pixel_decoder_upsampling_stages,
        )
        self.mask_predictor = Sam3MaskPredictor(config.d_model, config.d_model)
        self.instance_seg_head = nn.Conv2d(config.d_model, config.d_model, kernel_size=1)
        self.semantic_seg_head = nn.Conv2d(config.d_model, 1, kernel_size=1)

        self.cross_attend_prompt = Sam3CrossAttention(config.d_model, config.encoder_n_head)
        self.cross_attn_norm = LayerNorm(config.d_model)

        if config.presence_head:
            self.presence_head = nn.Sequential(
                nn.Identity(),
                nn.Identity(),
                Linear(config.d_model, 1, bias=True),
            )
        else:
            self.presence_head = None

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        obj_queries: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Predict segmentation masks.

        Args:
            backbone_feats: FPN feature maps.
            obj_queries: (B, Q, D) decoder hidden states.
            encoder_hidden_states: (L, B, D) encoder memory (seq-first).
            prompt: (S, B, D) text/prompt features (seq-first).
            prompt_mask: (B, S) padding mask.

        Returns:
            Dict with 'pred_masks', 'semantic_seg', 'presence_logit'.
        """
        if prompt is not None:
            tgt2 = self.cross_attn_norm(encoder_hidden_states)
            tgt2_bf = tgt2.transpose(0, 1)  # (B, L, D)
            prompt_bf = prompt.transpose(0, 1)  # (B, S, D)
            tgt2 = self.cross_attend_prompt(
                tgt2_bf, prompt_bf, prompt_bf,
                key_padding_mask=prompt_mask,
            )
            encoder_hidden_states = tgt2.transpose(0, 1) + encoder_hidden_states

        B = obj_queries.shape[0]

        spatial_dim = math.prod(backbone_feats[-1].shape[-2:])
        enc_vis = encoder_hidden_states.permute(1, 2, 0)  # (B, D, L)
        enc_vis = enc_vis[..., :spatial_dim].reshape(-1, *backbone_feats[-1].shape[1:])

        vis_feats = list(backbone_feats)
        vis_feats[-1] = enc_vis

        pixel_embed = self.pixel_decoder(vis_feats)
        instance_embeds = self.instance_seg_head(pixel_embed)
        mask_preds = self.mask_predictor(obj_queries, instance_embeds)

        presence_logit = None
        if self.presence_head is not None:
            pooled = encoder_hidden_states.mean(0)  # (B, D)
            presence_logit = self.presence_head(pooled)

        return {
            "pred_masks": mask_preds,
            "semantic_seg": self.semantic_seg_head(pixel_embed),
            "presence_logit": presence_logit,
        }


def load_sam3_checkpoint(model: "Sam3Model", checkpoint_path: str) -> Tuple[list, list]:
    """Load a reference SAM3 checkpoint into a kb-nano Sam3Model.

    Handles the key remapping between the reference Sam3Image module hierarchy
    and the kb-nano Sam3Model hierarchy, including:
    - Prefix remapping (backbone.vision_backbone -> neck, etc.)
    - FPN conv name remapping (named submodules -> sequential indices)
    - Fused in_proj_weight/bias splitting into separate q/k/v projections
    - Text encoder layer renaming
    - Encoder/decoder attention key splitting

    Args:
        model: kb-nano Sam3Model instance.
        checkpoint_path: Path to the .pt checkpoint file.

    Returns:
        (missing_keys, unexpected_keys) from load_state_dict.
    """
    import re

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    det_keys = {
        k.replace("detector.", ""): v
        for k, v in ckpt.items()
        if "detector" in k
    }

    # --- FPN conv key mapping ---
    # Reference: convs.0.dconv_2x2_0 / .dconv_2x2_1 / .conv_1x1 / .conv_3x3
    # kb-nano:   convs.0.conv.{0,1,2,...}  (via nn.Sequential)
    #
    # Scale 4.0 (convs.0): dconv_2x2_0->0, GELU->1(skipped), dconv_2x2_1->2, conv_1x1->3, conv_3x3->4
    # Scale 2.0 (convs.1): dconv_2x2->0, conv_1x1->1, conv_3x3->2
    # Scale 1.0 (convs.2): conv_1x1->0, conv_3x3->1
    # Scale 0.5 (convs.3): MaxPool2d->0(no params), conv_1x1->1, conv_3x3->2
    fpn_remap = {
        "convs.0.dconv_2x2_0": "convs.0.conv.0",
        "convs.0.dconv_2x2_1": "convs.0.conv.2",
        "convs.0.conv_1x1": "convs.0.conv.3",
        "convs.0.conv_3x3": "convs.0.conv.4",
        "convs.1.dconv_2x2": "convs.1.conv.0",
        "convs.1.conv_1x1": "convs.1.conv.1",
        "convs.1.conv_3x3": "convs.1.conv.2",
        "convs.2.conv_1x1": "convs.2.conv.0",
        "convs.2.conv_3x3": "convs.2.conv.1",
        "convs.3.conv_1x1": "convs.3.conv.1",
        "convs.3.conv_3x3": "convs.3.conv.2",
    }

    remapped = {}
    skipped_prefixes = (
        "geometry_encoder.", "dot_prod_scoring.",
        "inst_interactive_predictor.",
        "transformer.decoder.boxRPB_embed",
        "transformer.decoder.presence_token",
        "transformer.decoder.instance_",
        "transformer.encoder.text_pooling_proj.",
    )

    for ref_key, val in det_keys.items():
        if any(ref_key.startswith(p) for p in skipped_prefixes):
            continue

        new_key = ref_key

        # --- Top-level prefix remapping ---
        if new_key.startswith("backbone.vision_backbone."):
            rest = new_key[len("backbone.vision_backbone."):]
            # FPN conv remapping
            matched = False
            for ref_conv, kb_conv in fpn_remap.items():
                if rest.startswith(ref_conv):
                    suffix = rest[len(ref_conv):]
                    new_key = f"neck.{kb_conv}{suffix}"
                    matched = True
                    break
            if not matched:
                new_key = f"neck.{rest}"

        elif new_key.startswith("backbone.language_backbone.encoder."):
            rest = new_key[len("backbone.language_backbone.encoder."):]
            new_key = f"text_encoder.{rest}"

        elif new_key.startswith("backbone.language_backbone.resizer."):
            rest = new_key[len("backbone.language_backbone.resizer."):]
            new_key = f"text_encoder.resizer.{rest}"

        elif new_key.startswith("transformer.encoder."):
            rest = new_key[len("transformer.encoder."):]
            new_key = f"encoder.{rest}"

        elif new_key.startswith("transformer.decoder."):
            rest = new_key[len("transformer.decoder."):]
            new_key = f"decoder.{rest}"

        elif new_key.startswith("segmentation_head."):
            rest = new_key[len("segmentation_head."):]
            new_key = f"seg_head.{rest}"

        # --- Text encoder transformer layer remapping ---
        # Ref:  text_encoder.transformer.resblocks.{i}.attn.in_proj_weight
        # KB:   text_encoder.transformer_layers.{i}.block.q_proj.weight etc.
        m = re.match(r"text_encoder\.transformer\.resblocks\.(\d+)\.(.*)", new_key)
        if m:
            layer_idx = m.group(1)
            layer_rest = m.group(2)

            # Fused attention: in_proj_weight -> split q/k/v
            if layer_rest == "attn.in_proj_weight":
                d = val.shape[0] // 3
                remapped[f"text_encoder.transformer_layers.{layer_idx}.block.q_proj.weight"] = val[:d]
                remapped[f"text_encoder.transformer_layers.{layer_idx}.block.k_proj.weight"] = val[d:2*d]
                remapped[f"text_encoder.transformer_layers.{layer_idx}.block.v_proj.weight"] = val[2*d:]
                continue
            elif layer_rest == "attn.in_proj_bias":
                d = val.shape[0] // 3
                remapped[f"text_encoder.transformer_layers.{layer_idx}.block.q_proj.bias"] = val[:d]
                remapped[f"text_encoder.transformer_layers.{layer_idx}.block.k_proj.bias"] = val[d:2*d]
                remapped[f"text_encoder.transformer_layers.{layer_idx}.block.v_proj.bias"] = val[2*d:]
                continue
            elif layer_rest.startswith("attn.out_proj."):
                suffix = layer_rest[len("attn.out_proj."):]
                new_key = f"text_encoder.transformer_layers.{layer_idx}.block.out_proj.{suffix}"
            elif layer_rest.startswith("mlp.c_fc."):
                suffix = layer_rest[len("mlp.c_fc."):]
                new_key = f"text_encoder.transformer_layers.{layer_idx}.block.mlp_fc1.{suffix}"
            elif layer_rest.startswith("mlp.c_proj."):
                suffix = layer_rest[len("mlp.c_proj."):]
                new_key = f"text_encoder.transformer_layers.{layer_idx}.block.mlp_fc2.{suffix}"
            elif layer_rest.startswith("ln_1.") or layer_rest.startswith("ln_2."):
                new_key = f"text_encoder.transformer_layers.{layer_idx}.block.{layer_rest}"
            else:
                new_key = f"text_encoder.transformer_layers.{layer_idx}.block.{layer_rest}"

        # --- text_encoder token_embedding ---
        if new_key == "text_encoder.token_embedding.weight":
            new_key = "text_encoder.token_embedding.emb.weight"

        # --- text_encoder text_projection (not used in kb-nano, skip) ---
        if new_key == "text_encoder.text_projection":
            continue

        # --- Encoder layer: fused in_proj -> split q/k/v ---
        em = re.match(r"encoder\.layers\.(\d+)\.(self_attn|cross_attn_image)\.(.*)", new_key)
        if em:
            layer_idx = em.group(1)
            attn_name = "self_attn" if em.group(2) == "self_attn" else "cross_attn"
            attn_rest = em.group(3)

            if attn_rest == "in_proj_weight":
                d = val.shape[0] // 3
                remapped[f"encoder.layers.{layer_idx}.{attn_name}.q_proj.weight"] = val[:d]
                remapped[f"encoder.layers.{layer_idx}.{attn_name}.k_proj.weight"] = val[d:2*d]
                remapped[f"encoder.layers.{layer_idx}.{attn_name}.v_proj.weight"] = val[2*d:]
                continue
            elif attn_rest == "in_proj_bias":
                d = val.shape[0] // 3
                remapped[f"encoder.layers.{layer_idx}.{attn_name}.q_proj.bias"] = val[:d]
                remapped[f"encoder.layers.{layer_idx}.{attn_name}.k_proj.bias"] = val[d:2*d]
                remapped[f"encoder.layers.{layer_idx}.{attn_name}.v_proj.bias"] = val[2*d:]
                continue
            elif attn_rest.startswith("out_proj."):
                new_key = f"encoder.layers.{layer_idx}.{attn_name}.{attn_rest}"

        # --- Decoder layer: fused in_proj -> split q/k/v ---
        # Reference names: self_attn, cross_attn (image), ca_text
        # kb-nano names: self_attn, cross_attn (image), ca_text (same now)
        dm = re.match(r"decoder\.layers\.(\d+)\.(self_attn|cross_attn|ca_text)\.(.*)", new_key)
        if dm:
            layer_idx = dm.group(1)
            kb_attn = dm.group(2)
            attn_rest = dm.group(3)

            if attn_rest == "in_proj_weight":
                d = val.shape[0] // 3
                remapped[f"decoder.layers.{layer_idx}.{kb_attn}.q_proj.weight"] = val[:d]
                remapped[f"decoder.layers.{layer_idx}.{kb_attn}.k_proj.weight"] = val[d:2*d]
                remapped[f"decoder.layers.{layer_idx}.{kb_attn}.v_proj.weight"] = val[2*d:]
                continue
            elif attn_rest == "in_proj_bias":
                d = val.shape[0] // 3
                remapped[f"decoder.layers.{layer_idx}.{kb_attn}.q_proj.bias"] = val[:d]
                remapped[f"decoder.layers.{layer_idx}.{kb_attn}.k_proj.bias"] = val[d:2*d]
                remapped[f"decoder.layers.{layer_idx}.{kb_attn}.v_proj.bias"] = val[2*d:]
                continue
            elif attn_rest.startswith("out_proj."):
                new_key = f"decoder.layers.{layer_idx}.{kb_attn}.{attn_rest}"

        # --- Decoder bbox_embed: single shared MLP ---
        # Reference: bbox_embed.layers.{0,1,2} → kb-nano: bbox_embed.{0,2,4}
        dm_bbox = re.match(r"decoder\.bbox_embed\.layers\.(\d+)\.(.*)", new_key)
        if dm_bbox:
            lin_idx = int(dm_bbox.group(1))
            suffix = dm_bbox.group(2)
            seq_idx = lin_idx * 2  # 0->0, 1->2, 2->4
            new_key = f"decoder.bbox_embed.{seq_idx}.{suffix}"

        # --- Decoder ref_point_head: layers.{i} -> {i*2} ---
        dm_rph = re.match(r"decoder\.ref_point_head\.layers\.(\d+)\.(.*)", new_key)
        if dm_rph:
            lin_idx = int(dm_rph.group(1))
            suffix = dm_rph.group(2)
            seq_idx = lin_idx * 2
            new_key = f"decoder.ref_point_head.{seq_idx}.{suffix}"

        # --- Segmentation head cross_attend_prompt: fused in_proj -> split q/k/v ---
        if new_key == "seg_head.cross_attend_prompt.in_proj_weight":
            d = val.shape[0] // 3
            remapped["seg_head.cross_attend_prompt.q_proj.weight"] = val[:d]
            remapped["seg_head.cross_attend_prompt.k_proj.weight"] = val[d:2*d]
            remapped["seg_head.cross_attend_prompt.v_proj.weight"] = val[2*d:]
            continue
        if new_key == "seg_head.cross_attend_prompt.in_proj_bias":
            d = val.shape[0] // 3
            remapped["seg_head.cross_attend_prompt.q_proj.bias"] = val[:d]
            remapped["seg_head.cross_attend_prompt.k_proj.bias"] = val[d:2*d]
            remapped["seg_head.cross_attend_prompt.v_proj.bias"] = val[2*d:]
            continue

        # --- Segmentation head mask_predictor: mask_embed.layers -> layers ---
        sm = re.match(r"seg_head\.mask_predictor\.mask_embed\.layers\.(\d+)\.(.*)", new_key)
        if sm:
            new_key = f"seg_head.mask_predictor.layers.{sm.group(1)}.{sm.group(2)}"

        # --- ViT block attention freqs_cis: trunk.blocks.{i}.attn.freqs_cis ---
        if ".attn.freqs_cis" in new_key:
            new_key = new_key.replace(".attn.freqs_cis", ".attn.rope.freqs_cis")

        # --- ViT MLP: mlp.fc1/fc2 -> mlp.fc1/fc2 (same in kb-nano) ---
        # (already matches)

        remapped[new_key] = val

    # Handle kb-nano Linear wrapper: weight is stored directly, not inside .matmul
    # Our Linear has .weight and .bias directly as nn.Parameter, same as nn.Linear
    # But our L1 Linear wraps Matmul - check if state dict keys need adjustment
    # Actually, our Linear stores .weight and .bias as nn.Parameter + .matmul as Matmul
    # The state dict key is "xxx.weight" not "xxx.matmul.weight", so it should work.

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    return missing, unexpected


class Sam3Model(nn.Module):
    """Full SAM3 image segmentation model.

    Assembles VL backbone (ViT + neck + text encoder), fusion encoder,
    detection decoder, and segmentation head.

    Args:
        config: Sam3Config with all hyperparameters.
    """

    def __init__(self, config: Sam3Config):
        super().__init__()
        self.config = config

        vit = Sam3ViT(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.vit_depth,
            num_heads=config.vit_num_heads,
            mlp_ratio=config.vit_mlp_ratio,
            window_size=config.window_size,
            global_att_blocks=config.global_att_blocks,
            use_rope=config.use_rope,
            use_tiled_rope=config.use_tiled_rope,
            pretrain_img_size=config.pretrain_img_size,
            retain_cls_token=config.retain_cls_token,
            ln_pre=config.ln_pre,
            bias_patch_embed=config.bias_patch_embed,
            return_interm_layers=False,
        )

        self.neck = Sam3Neck(
            trunk=vit,
            d_model=config.d_model,
            scale_factors=config.scale_factors,
        )

        self.text_encoder = Sam3TextEncoder(
            d_model=config.d_model,
            width=config.text_width,
            heads=config.text_heads,
            layers=config.text_layers,
            context_length=config.text_context_length,
            vocab_size=config.text_vocab_size,
        )

        self.encoder = Sam3FusionEncoder(config)
        self.decoder = Sam3Decoder(config)
        self.seg_head = Sam3SegmentationHead(config)

        self.scalp = config.scalp
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal (std=0.02) for stability."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, Linear)):
                w = m.weight if hasattr(m, "weight") else None
                if w is not None:
                    nn.init.trunc_normal_(w, std=0.02)
                b = m.bias if hasattr(m, "bias") else None
                if b is not None:
                    nn.init.zeros_(b)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, LayerNorm):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        images: torch.Tensor,
        tokenized_text: torch.Tensor,
    ) -> Dict[str, Any]:
        """Run full SAM3 forward pass.

        Args:
            images: (B, 3, H, W) input images, normalized.
            tokenized_text: (B, seq_len) tokenized text prompts.

        Returns:
            Dict with:
                'pred_boxes': (B, Q, 4) predicted bounding boxes.
                'pred_masks': (B, Q, H, W) predicted segmentation masks.
                'pred_logits': (B, Q, 1) or None, presence logits.
        """
        sam3_feats, sam3_pos, _, _ = self.neck(images)

        if self.scalp > 0:
            sam3_feats = sam3_feats[:-self.scalp]
            sam3_pos = sam3_pos[:-self.scalp]

        text_mask, text_memory, text_embeds = self.text_encoder(tokenized_text)

        encoder_out = self.encoder(
            src=sam3_feats,
            src_pos=sam3_pos,
            prompt=text_memory,
            prompt_key_padding_mask=text_mask,
        )

        hs, pred_boxes = self.decoder(
            memory=encoder_out["memory"],
            pos_embed=encoder_out["pos_embed"],
            memory_text=encoder_out["memory_text"],
            text_attention_mask=text_mask,
        )

        seg_out = self.seg_head(
            backbone_feats=sam3_feats,
            obj_queries=hs,
            encoder_hidden_states=encoder_out["memory"],
            prompt=encoder_out["memory_text"],
            prompt_mask=text_mask,
        )

        return {
            "pred_boxes": pred_boxes,
            "pred_masks": seg_out["pred_masks"],
            "pred_logits": seg_out.get("presence_logit"),
            "semantic_seg": seg_out.get("semantic_seg"),
        }
