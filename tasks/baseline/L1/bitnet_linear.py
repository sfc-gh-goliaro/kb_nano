"""BitLinear: linear layer for BitNet b1.58 with W1.58A8 inference.

Implements the "Native 1.58-bit weights and 8-bit activations" (W1.58A8)
format used by Microsoft's ``microsoft/bitnet-b1.58-2B-4T`` checkpoint:

    w_quant   in {-1, 0, +1}      (per-tensor scale ``weight_scale``)
    a_quant   per-token int8      (per-row scale ``act_scale``)
    y         = (a_quant @ w_quant.T) * weight_scale / act_scale + bias

The HuggingFace checkpoint stores ``weight`` in one of two formats:

    * **offline / packed**:  uint8 tensor of shape ``(out//4, in)`` where each
      byte packs four ternary values (2 bits each, biased by +1 so unsigned).
      ``weight_scale`` is a scalar bf16 stored alongside.
    * **online / bf16**:     bf16 tensor of shape ``(out, in)`` already
      containing dequantized ternary values (master weights).  In this mode
      the model card omits ``weight_scale`` and the scale is implicit (=1).

After loading, ``self.weight`` always holds the unpacked bf16 ternary tensor
``(out, in)`` and ``self.weight_scale`` is a scalar bf16 buffer.  Forward
matches HuggingFace's ``AutoBitLinear`` (offline mode) and Microsoft's
reference ``BitLinear``: per-token int8 activation quantization, dequantized
matmul, then multiplied by ``weight_scale``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Two-bit ternary packing factor used by the offline checkpoint format.
VALUES_PER_ITEM = 4


def unpack_ternary_weights(packed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Unpack a uint8 (out//4, in) tensor into bf16 ternary (out, in).

    Layout matches ``transformers.integrations.bitnet.unpack_weights``: each
    byte stores four 2-bit values along the row dimension; the i-th value
    occupies bits ``2i .. 2i+1`` and is biased by +1 (so 0 -> -1, 1 -> 0,
    2 -> +1).
    """
    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)
    packed_rows = packed.shape[0]
    out_rows = packed_rows * VALUES_PER_ITEM
    rest = packed.shape[1:]

    unpacked = torch.empty((out_rows, *rest), device=packed.device, dtype=torch.uint8)
    for i in range(VALUES_PER_ITEM):
        mask = 3 << (2 * i)
        unpacked[i * packed_rows:(i + 1) * packed_rows] = (packed & mask) >> (2 * i)
    return unpacked.to(dtype) - 1


def activation_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token int8 activation quantization.

    Returns ``(x_int8_as_dtype, scale)`` where ``scale`` is the per-token
    factor such that ``x_int8 / scale`` recovers the dequantized activation.
    The returned tensor is kept in the input dtype (rather than int8) to
    feed straight into ``F.linear`` without an extra cast.
    """
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5)
    q = (x * scale).round().clamp_(-128, 127)
    return q, scale


class BitLinear(nn.Module):
    """W1.58A8 linear layer.

    ``weight`` and ``weight_scale`` are exposed as ``nn.Parameter`` so they
    integrate with the shared :func:`~kb_nano.infra.weight_loader.load_weights`
    pipeline (which uses ``model.get_parameter`` to locate each tensor).
    Both are non-trainable.  The custom ``weight_loader`` callback on
    ``weight`` transparently unpacks the packed uint8 (out//4, in) format
    used by the offline HuggingFace checkpoint into the bf16 (out, in)
    ternary tensor expected at runtime.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if dtype is None:
            dtype = torch.get_default_dtype()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.weight.weight_loader = self._weight_loader
        self.weight_scale = nn.Parameter(
            torch.ones(1, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.weight_scale.weight_loader = self._scale_loader
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self.bias = None

    def _weight_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        if loaded.dtype == torch.uint8 or (
            loaded.shape[0] * VALUES_PER_ITEM == param.data.shape[0]
            and loaded.shape[1] == param.data.shape[1]
        ):
            loaded = unpack_ternary_weights(loaded, dtype=param.data.dtype)
        param.data.copy_(loaded)

    def _scale_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        if loaded.numel() == param.data.numel():
            param.data.copy_(loaded.reshape(param.data.shape))
        else:
            param.data.copy_(loaded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q, act_scale = activation_quant(x)
        y = F.linear(x_q, self.weight)
        y = y * (self.weight_scale / act_scale)
        if self.bias is not None:
            y = y + self.bias
        return y
