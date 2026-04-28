"""Semantic PyTorch reference for fused_recurrent_retention.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kb_nano.tasks.reference.L1.gla_recurrence import naive_recurrent_gla


class FusedRecurrentRetention(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del cu_seqlens
        heads = q.shape[2]
        h_idx = torch.arange(heads, dtype=torch.float32, device=q.device)
        gamma = 1.0 - torch.pow(torch.tensor(2.0, dtype=torch.float32, device=q.device), -5.0 - h_idx)
        gk = torch.log(gamma).to(q.dtype).view(1, 1, heads, 1).expand_as(q)
        out, state = naive_recurrent_gla(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            gk.transpose(1, 2),
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        return out.transpose(1, 2), state
