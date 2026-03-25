"""Linear (matrix multiply) kernels.

Matmul: pure functional op — F.linear(input, weight, bias).
Linear: parametric op — holds weight/bias as nn.Parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Matmul(nn.Module):
    """Pure functional linear: takes input, weight, and optional bias as forward args."""

    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)


class Linear(nn.Module):
    """Parametric linear: stores weight and bias internally."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.op = Matmul()

    def forward(self, input):
        return self.op(input, self.weight, self.bias)
