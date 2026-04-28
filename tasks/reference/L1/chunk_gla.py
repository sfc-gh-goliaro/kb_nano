"""Semantic PyTorch reference for chunk_gla.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kb_nano.tasks.reference.L1.gla_recurrence import naive_recurrent_gla


class ChunkGLA(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del cu_seqlens
        out, state = naive_recurrent_gla(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            g.transpose(1, 2),
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        return out.transpose(1, 2), state
