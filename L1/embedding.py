"""Embedding lookup kernel: F.embedding(input, weight, ...)."""

import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def forward(self, input, weight):
        return F.embedding(input, weight)
