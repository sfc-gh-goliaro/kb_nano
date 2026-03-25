"""Minimal VAE image post-processor (L4 utility).

Handles the tensor -> PIL conversion after VAE decoding: denormalize
from [-1, 1] to [0, 1], convert to numpy, then to PIL images.

Mirrors the ``postprocess`` path of diffusers' ``VaeImageProcessor``.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class VaeImageProcessor:
    """Lightweight image post-processor for VAE output tensors."""

    def __init__(self, vae_scale_factor: int = 8, do_normalize: bool = True):
        self.vae_scale_factor = vae_scale_factor
        self.do_normalize = do_normalize

    def postprocess(
        self,
        image: torch.Tensor,
        output_type: str = "pil",
    ) -> list[Image.Image] | np.ndarray | torch.Tensor:
        if output_type == "latent" or output_type == "pt":
            return image

        if self.do_normalize:
            image = (image * 0.5 + 0.5).clamp(0, 1)

        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "np":
            return image

        images = (image * 255).round().astype("uint8")
        return [Image.fromarray(img) for img in images]
