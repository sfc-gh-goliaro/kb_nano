"""Standard LayerNorm wrapping F.layer_norm with optional affine parameters.

Supports create_scale and create_offset flags matching the reference
openfold3/core/model/primitives/normalization.py LayerNorm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool | None = None,
        create_scale: bool = True,
        create_offset: bool = True,
    ):
        """LayerNorm wrapping F.layer_norm.

        ``bias`` is the torch.nn.LayerNorm-compatible kwarg (HF models like
        bark/dbrx/gemma4/modernbert/moonshine pass ``bias=False`` /
        ``bias=config.norm_bias``). If provided, it overrides ``create_offset``
        for compatibility. ``create_scale`` / ``create_offset`` are preserved
        for openfold3 callers.
        """
        super().__init__()
        # Accept int or tuple/list (torch.nn.LayerNorm semantics)
        if isinstance(normalized_shape, int):
            self.normalized_shape: tuple[int, ...] = (normalized_shape,)
            shape_for_params = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
            shape_for_params = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # Resolve bias-related flags. torch.nn.LayerNorm's `bias` kwarg, when
        # passed, takes precedence over our openfold3-style `create_offset`.
        if bias is not None:
            create_offset = bool(bias)

        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(*shape_for_params))
        else:
            self.register_parameter("weight", None)

        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(*shape_for_params))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Promote to fp32 for the reduction to match vLLM's
        # ``vllm/model_executor/layers/layernorm.py:LayerNorm`` which keeps
        # ``weight`` / ``bias`` in fp32 and runs the reduction in fp32.
        # Matters for the DeepSeek-V3.2 indexer ``k_norm`` — running the
        # reduction in bf16 biases the variance enough to shift the
        # FP8-quantized indexer K cache, which in turn changes the top-2048
        # selection in every sparse layer.
        orig_dtype = x.dtype
        weight = self.weight
        bias = self.bias
        if weight is not None and weight.dtype != torch.float32:
            weight = weight.float()
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.float()
        return F.layer_norm(
            x.float(), self.normalized_shape, weight, bias, self.eps,
        ).to(orig_dtype)
