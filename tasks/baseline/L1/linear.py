"""Linear (matrix multiply) kernel: F.linear(input, weight, bias)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """When constructed with dimensions, holds weight/bias parameters.
    Otherwise acts as a pure functional op."""

    def __init__(self, in_features: int = 0, out_features: int = 0, bias: bool = True):
        super().__init__()
        if in_features > 0 and out_features > 0:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        else:
            self.weight = None
            self.bias = None

    def forward(self, input, weight=None, bias=None):
        w = weight if weight is not None else self.weight
        b = bias if bias is not None else self.bias
        return F.linear(input, w, b)
