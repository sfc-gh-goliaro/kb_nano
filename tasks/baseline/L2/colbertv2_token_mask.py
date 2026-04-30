"""ColBERT token masking."""

from __future__ import annotations

import torch
import torch.nn as nn


class ColBERTv2TokenMask(nn.Module):
    def __init__(self, pad_token_id: int):
        super().__init__()
        self.pad_token_id = pad_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        skiplist: set[int] | None = None,
    ) -> torch.Tensor:
        blocked = {self.pad_token_id}
        if skiplist:
            blocked |= {int(token_id) for token_id in skiplist}
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        for token_id in blocked:
            mask &= input_ids.ne(token_id)
        return mask
