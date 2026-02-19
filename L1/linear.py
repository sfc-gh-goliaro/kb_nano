"""Linear (matrix multiply) kernel: F.linear(input, weight, bias)."""

import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)
