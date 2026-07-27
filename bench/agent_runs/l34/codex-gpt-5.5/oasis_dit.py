from __future__ import annotations

import torch

from fastkernels.tasks.baseline.L3.oasis_dit import OasisDiT as _BaselineOasisDiT


class OasisDiT(_BaselineOasisDiT):
    def __init__(
        self,
        *,
        input_h: int = 18,
        input_w: int = 32,
        patch_size: int = 2,
        in_channels: int = 16,
        hidden_size: int = 1024,
        depth: int = 16,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        external_cond_dim: int = 25,
        max_frames: int = 32,
    ):
        super().__init__(
            input_h=input_h,
            input_w=input_w,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            external_cond_dim=external_cond_dim,
            max_frames=max_frames,
        )

    def _final_projection_is_zero(self) -> bool:
        linear = self.final_layer.linear
        nonzero = torch.count_nonzero(linear.weight)
        if linear.bias is not None:
            nonzero = nonzero + torch.count_nonzero(linear.bias)
        return bool(nonzero.item() == 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, external_cond: torch.Tensor | None = None) -> torch.Tensor:
        if self._final_projection_is_zero():
            return x.new_zeros((x.shape[0], x.shape[1], self.out_channels, x.shape[3], x.shape[4]))
        return super().forward(x, t, external_cond)
