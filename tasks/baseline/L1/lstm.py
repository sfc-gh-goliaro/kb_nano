"""LSTM L1 op — subclass alias of ``torch.nn.LSTM`` (no kb-nano kernel).

**HONEST CAVEAT.** This file adds zero compute functionality. ``LSTM(...)``
constructs an object byte-for-byte identical to ``nn.LSTM(...)``: same
parameters, same state_dict keys (``weight_ih_l{k}`` / ``weight_hh_l{k}``
/ ``bias_ih_l{k}`` / ``bias_hh_l{k}`` + ``_reverse`` variants), same
forward kernel (ATen / cuDNN). kb-nano never authored an LSTM kernel.

Why the file exists at all:
1. CLAUDE.md "L2+ must use only L1 ops" — L2 callers that need an LSTM
   can ``from ..L1.lstm import LSTM`` instead of importing from torch.nn,
   keeping them L1-only.
2. Single canonical-op-per-file convention so the bench catalog has an
   ``lstm`` entry that resolves.

If kb-nano ever ships a real LSTM kernel (chunked / fused / lower-precision
cuDNN replacement), it would land here.

The previous implementation wrapped ``nn.LSTM`` as ``self.lstm`` (composition).
That broke state_dict-compat with HF checkpoints (keys nested under ``lstm.``).
The subclass form preserves bare keys.
"""

from __future__ import annotations

import torch.nn as nn


class LSTM(nn.LSTM):
    """Subclass alias of ``torch.nn.LSTM``. No kb-nano kernel; pure cuDNN."""
    pass
