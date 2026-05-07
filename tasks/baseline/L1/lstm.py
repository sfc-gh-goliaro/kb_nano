"""LSTM L1 op — minimal nn.Module wrapping ``torch.nn.LSTM`` (no kb-nano kernel).

**HONEST CAVEAT.** This file adds zero compute functionality. The forward
kernel is provided entirely by ATen / cuDNN. kb-nano never authored an LSTM
kernel. This wrapper exists for L1-catalog conformance per the kb-nano rule
that every L1 op subclasses ``nn.Module`` directly (NOT another nn class).

Implementation: minimal forwarding wrapper around an internal ``nn.LSTM``
instance. The state_dict keys are nested under ``lstm.`` (e.g.
``lstm.weight_ih_l0``); call sites loading an HF checkpoint must remap
or do ``kb.lstm.load_state_dict(ref.state_dict())``.

If kb-nano ever ships a real LSTM kernel (chunked / fused / lower-precision
cuDNN replacement), it would land here as the actual forward implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """Minimal ``nn.Module`` wrapper around ``torch.nn.LSTM``."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.lstm(*args, **kwargs)
