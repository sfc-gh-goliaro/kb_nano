"""SigLIP-2 NaFlexVit SO400M vision encoder (L4).

Full Vision Transformer matching timm's NaFlexVit for the SigLIP-2
SO400M/16 variant. Supports standard fixed-resolution image input
and outputs pooled 1152-D feature embeddings via MAP (multi-head
attention pooling).

Reference: timm/models/naflexvit.py NaFlexVit
           timm model name: naflexvit_so400m_patch16_siglip.v2_webli
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.layer_norm import LayerNorm
from ..L2.attention_pool import AttentionPoolLatent
from ..L3.vit_encoder_block import VitEncoderBlock


SIGLIP2_SO400M_CONFIG = dict(
    patch_size=16,
    embed_dim=1152,
    depth=27,
    num_heads=16,
    mlp_ratio=4304 / 1152,
    act_approximate="tanh",
    default_resolution=384,
)

HF_TO_TIMM = {
    "google/siglip2-so400m-patch16-naflex": "naflexvit_so400m_patch16_siglip.v2_webli",
}


class SigLIP2Model(nn.Module):
    """SigLIP-2 NaFlexVit SO400M/16 vision encoder.

    Args:
        patch_size: Patch size.
        in_chans: Input image channels.
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Attention heads.
        mlp_ratio: MLP expansion ratio.
        act_approximate: GELU approximation ("none" or "tanh").
        pos_embed_grid_size: Grid size for learned position embedding.
        num_classes: Output classes (0 = feature extractor only).
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1152,
        depth: int = 27,
        num_heads: int = 16,
        mlp_ratio: float = 3.7362,
        act_approximate: str = "tanh",
        pos_embed_grid_size: Tuple[int, int] = (16, 16),
        num_classes: int = 0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_prefix_tokens = 0

        # Patch embedding (linear projection for pre-patchified inputs, matching timm NaFlexVit)
        patch_dim = patch_size * patch_size * in_chans
        self.proj = nn.Linear(patch_dim, embed_dim, bias=True)

        # Learned position embedding: (1, H, W, C) matching timm NaFlex format
        gh, gw = pos_embed_grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, gh, gw, embed_dim))

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            VitEncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                act_approximate=act_approximate,
            )
            for _ in range(depth)
        ])

        # Final norm
        self.norm = LayerNorm(embed_dim)

        # MAP attention pooling (uses model's mlp_ratio, not fixed 4.0)
        self.attn_pool = AttentionPoolLatent(
            in_features=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            pool_type="token",
        )

        # Classifier head
        self.fc_norm = nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _interpolate_pos_embed(
        self,
        grid_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Interpolate position embedding to match the input grid size."""
        pos = self.pos_embed  # (1, Hg, Wg, C)
        gh, gw = pos.shape[1], pos.shape[2]
        th, tw = grid_size
        if (gh, gw) == (th, tw):
            return pos.reshape(1, gh * gw, -1)
        pos = pos.permute(0, 3, 1, 2)  # (1, C, Hg, Wg)
        pos = F.interpolate(pos, size=(th, tw), mode="bicubic", align_corners=False)
        pos = pos.permute(0, 2, 3, 1)  # (1, H, W, C)
        return pos.reshape(1, th * tw, -1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a standard (B, C, H, W) image tensor."""
        B, C, H, W = x.shape
        ph = pw = self.patch_size
        gh, gw = H // ph, W // pw

        # Patchify: (B, C, H, W) -> (B, N, patch_dim)
        x = x.reshape(B, C, gh, ph, gw, pw)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, gh * gw, ph * pw * C)

        # Linear projection
        x = self.proj(x)

        # Add interpolated position embedding
        pos = self._interpolate_pos_embed((gh, gw))
        x = x + pos

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Final norm
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_pool(x)
        x = self.fc_norm(x)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @staticmethod
    def from_timm(model_name: str = "naflexvit_so400m_patch16_siglip.v2_webli") -> SigLIP2Model:
        """Load weights from a timm pretrained checkpoint.

        Handles the key remapping between timm's NaFlexVit weight names
        (embeds.proj, embeds.pos_embed, blocks.N.attn.qkv, etc.) and
        this module's weight names.
        """
        import timm

        timm_model = timm.create_model(model_name, pretrained=True)
        timm_sd = timm_model.state_dict()

        cfg = SIGLIP2_SO400M_CONFIG
        pos_grid = timm_model.embeds.pos_embed.shape[1:3] if hasattr(timm_model, "embeds") else (16, 16)

        kb_model = SigLIP2Model(
            patch_size=cfg["patch_size"],
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            act_approximate=cfg["act_approximate"],
            pos_embed_grid_size=pos_grid,
            num_classes=0,
        )

        new_sd = _remap_timm_to_kb(timm_sd)

        missing, unexpected = kb_model.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"  SigLIP2 load: {len(missing)} missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  SigLIP2 load: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

        del timm_model
        return kb_model


def _remap_timm_to_kb(timm_sd: dict) -> dict:
    """Remap timm NaFlexVit state dict keys to kb-nano SigLIP2Model keys.

    timm keys (after checkpoint_filter_fn):
        embeds.proj.weight          -> proj.weight
        embeds.proj.bias            -> proj.bias
        embeds.pos_embed            -> pos_embed
        blocks.0.norm1.weight       -> blocks.0.norm1.weight
        blocks.0.attn.qkv.weight    -> blocks.0.attn.qkv.weight (Linear wraps Matmul)
        blocks.0.attn.proj.weight   -> blocks.0.attn.proj.weight
        blocks.0.mlp.fc1.weight     -> blocks.0.mlp.fc1.weight
        blocks.0.mlp.fc2.weight     -> blocks.0.mlp.fc2.weight
        norm.weight                 -> norm.weight
        attn_pool.*                 -> attn_pool.*
        fc_norm.*                   -> fc_norm.*
        head.*                      -> head.*
    """
    out = {}
    for k, v in timm_sd.items():
        nk = k
        # embeds.proj -> proj
        if k.startswith("embeds.proj."):
            nk = k.replace("embeds.proj.", "proj.")
        elif k.startswith("embeds.pos_embed"):
            nk = k.replace("embeds.", "")
        elif k.startswith("embeds."):
            nk = k.replace("embeds.", "")

        # Remap nn.Linear weights to kb_nano Linear (which wraps Matmul)
        nk = _remap_linear_key(nk)

        out[nk] = v
    return out


def _remap_linear_key(key: str) -> str:
    """Remap plain linear weight/bias to kb_nano Linear structure.

    timm: module.weight / module.bias
    kb_nano Linear: module.weight / module.bias  (same, stored as nn.Parameter)

    For the L1 Linear wrapper, weights are stored directly as self.weight
    and self.bias (nn.Parameter), same names as nn.Linear. No remapping
    needed for parameter names, only for nested Matmul which has no params.
    """
    return key
