"""FP8 Linear operation: drop-in replacement for L1/linear.py's Linear.

Combines per-token-group FP8 activation quantization with block-scaled
FP8 GEMM. Weights are already stored as float8_e4m3fn with pre-computed
per-block scale factors.
"""

import torch
import torch.nn as nn

from .fp8_quant import PerTokenGroupQuantFP8
from .fp8_block_scaled_mm import W8A8BlockScaledMM


class FP8Linear(nn.Module):
    """FP8 W8A8 block-quantized linear operation.

    Args:
        block_size: (block_n, block_k) quantization block dimensions,
                    e.g. (128, 128).

    forward(input, weight, weight_scale_inv, bias=None, input_scale=None) -> output:
        input:            [*, K] in BF16 (or float8_e4m3fn if input_scale given)
        weight:           [N, K] in float8_e4m3fn
        weight_scale_inv: [ceil(N/bn), ceil(K/bk)] in float32
        bias:             [N] optional
        input_scale:      [M, ceil(K/bk)] optional pre-computed activation scales;
                          when None, activations are dynamically quantized.
        output:           [*, N] in BF16
    """

    def __init__(self, block_size: tuple[int, int] = (128, 128)):
        super().__init__()
        self.block_size = block_size
        self.quant = PerTokenGroupQuantFP8(group_size=block_size[1])
        self.mm = W8A8BlockScaledMM()

    def forward(self, input, weight, weight_scale_inv, bias=None,
                input_scale=None):
        input_shape = input.shape
        input_2d = input.view(-1, input_shape[-1])

        if input_scale is not None:
            input_fp8 = input_2d
        else:
            input_fp8, input_scale = self.quant(input_2d)
        output = self.mm(input_fp8, weight, input_scale, weight_scale_inv, self.block_size)

        if bias is not None:
            output = output + bias

        return output.view(*input_shape[:-1], weight.shape[0])
