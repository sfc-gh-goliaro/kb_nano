"""Top-k selection kernel: torch.topk(input, k, ...)."""

import torch
import torch.nn as nn


class Topk(nn.Module):
    def forward(self, input, k, dim=-1, largest=True, sorted=True):
        return torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)
