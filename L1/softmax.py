"""Softmax kernel: F.softmax(input, dim, dtype)."""

import torch.nn as nn
import torch.nn.functional as F


class Softmax(nn.Module):
    def forward(self, input, dim, dtype=None):
        return F.softmax(input, dim=dim, dtype=dtype)
