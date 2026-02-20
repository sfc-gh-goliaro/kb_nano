"""Mixtral Mixture-of-Experts block with fused Triton grouped GEMM."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from ...infra.tp import _tp_rank, _tp_size, get_custom_ar
from ..L1.softmax import Softmax
from ..L1.topk import Topk
from ..L2.fused_experts import FusedExperts


class MixtralMoE(nn.Module):
    """Mixture-of-Experts with fused Triton grouped GEMM.

    Weights:
      w13: [E, 2*intermediate_per_tp, hidden_size] -- gate (w1) and up (w3) stacked
      w2:  [E, hidden_size, intermediate_per_tp]
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.intermediate_size // tp

        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.gate.weight.weight_loader = lambda p, w: p.data.copy_(w)

        self.w13 = nn.Parameter(torch.empty(
            config.num_local_experts, 2 * self.intermediate_per_tp, config.hidden_size,
        ))
        self.w13.weight_loader = self._w13_weight_loader

        self.w2 = nn.Parameter(torch.empty(
            config.num_local_experts, config.hidden_size, self.intermediate_per_tp,
        ))
        self.w2.weight_loader = self._w2_weight_loader

        self.softmax_op = Softmax()
        self.topk_op = Topk()
        self.fused_experts = FusedExperts()

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * N, N)
        offset = 0 if is_w1 else N
        param.data[expert_id, offset:offset + N, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * N, N))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits = self.gate(hidden_states)
        routing_weights = self.softmax_op(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = self.topk_op(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_states.dtype)

        out = self.fused_experts(
            hidden_states, self.w13, self.w2,
            topk_weights, topk_ids, self.num_experts,
        )

        if self.tp_size > 1:
            ar = get_custom_ar()
            if ar is not None:
                reduced = ar.custom_all_reduce(out)
                if reduced is not None:
                    return reduced.view(orig_shape)
            dist.all_reduce(out)

        return out.view(orig_shape)
