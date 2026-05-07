"""Oasis 2D patch embedding."""

from __future__ import annotations

from pathlib import Path
import sys
_L2_DIR = Path(__file__).resolve().parent
_L1_DIR = _L2_DIR.parent / "L1"
for _p in (str(_L2_DIR), str(_L1_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import torch
import torch.nn as nn

from conv2d import Conv2d


class OasisPatchEmbed(nn.Module):
    def __init__(
        self,
        img_height: int = 256,
        img_width: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        flatten: bool = True,
    ):
        super().__init__()
        self.img_size = (img_height, img_width)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_height // patch_size, img_width // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x: torch.Tensor, random_sample: bool = False) -> torch.Tensor:
        _, _, height, width = x.shape
        if not random_sample and (height, width) != self.img_size:
            raise AssertionError(
                f"Input image size ({height}*{width}) doesn't match model {self.img_size}.",
            )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        else:
            x = x.permute(0, 2, 3, 1)
        return self.norm(x) if self.norm is not None else x
