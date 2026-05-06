"""LSTM wrapping torch.nn.LSTM.

Mirrors the ``torch.nn.LSTM`` interface exactly so reference checkpoints
load with no remapping. Internally holds an ``nn.LSTM`` instance — kb-nano
exposes it as an L1 op for clarity (the LSTM kernel itself is provided by
ATen / cuDNN).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """Standard LSTM wrapping ``torch.nn.LSTM``."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
        )

    def forward(
        self,
        input: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.lstm(input, hx)
