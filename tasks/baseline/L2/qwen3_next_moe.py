"""Qwen3-Next MoE: softmax-gated with shared expert and sigmoid shared gate.

512 routed experts (top-10, softmax, renormalize), 1 shared expert with
sigmoid gate on the shared expert output.

Weight names match HuggingFace checkpoint:
  mlp.gate.weight                            [num_experts, hidden_size]
  mlp.shared_expert_gate.weight              [1, hidden_size]
  mlp.shared_expert.gate_proj.weight         [shared_intermediate, hidden_size]
  mlp.shared_expert.up_proj.weight           [shared_intermediate, hidden_size]
  mlp.shared_expert.down_proj.weight         [hidden_size, shared_intermediate]
  mlp.experts.{j}.gate_proj.weight           [moe_intermediate, hidden_size]
  mlp.experts.{j}.up_proj.weight             [moe_intermediate, hidden_size]
  mlp.experts.{j}.down_proj.weight           [hidden_size, moe_intermediate]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L2.fused_experts import FusedExperts
from .parallel_linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)


class _ReplicatedLinear(nn.Module):
    """Linear layer replicated across TP ranks (no sharding)."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = lambda p, w: p.data.copy_(w)
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class _SharedExpertMLP(nn.Module):
    """SiLU-gated MLP for shared expert."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size]
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        half = gate_up.shape[-1] // 2
        gate, up = gate_up[..., :half], gate_up[..., half:]
        x = torch.nn.functional.silu(gate) * up
        return self.down_proj(x)


class Qwen3NextMoE(nn.Module):
    """MoE with softmax gating, shared expert, and sigmoid shared gate."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.renormalize = config.norm_topk_prob
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.moe_intermediate_size // tp

        # Router (replicated)
        self.gate = _ReplicatedLinear(config.hidden_size, config.num_experts)

        # Shared expert sigmoid gate (replicated)
        self.shared_expert_gate = _ReplicatedLinear(config.hidden_size, 1)

        # Shared expert MLP
        self.shared_expert = _SharedExpertMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
        )

        # Fused routed expert weights: w13 = [gate, up] stacked, w2 = down
        N = self.intermediate_per_tp
        self.w13 = nn.Parameter(torch.empty(
            config.num_experts, 2 * N, config.hidden_size,
        ))
        self.w13.weight_loader = self._w13_weight_loader
        self.w2 = nn.Parameter(torch.empty(
            config.num_experts, config.hidden_size, N,
        ))
        self.w2.weight_loader = self._w2_weight_loader

        self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int, is_gate: bool):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * N, N)
        offset = 0 if is_gate else N
        param.data[expert_id, offset:offset + N, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * N, N))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        M = hidden_states.size(0)

        # Shared expert
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
        shared_output = self.shared_expert(hidden_states) * shared_gate

        # Routing: softmax top-k
        router_logits = self.gate(hidden_states)  # [M, E]
        scores = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = scores.topk(self.top_k, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights.to(hidden_states.dtype)
        topk_ids = topk_ids.to(torch.int32)

        # Routed experts
        routed_output = self.fused_experts(
            hidden_states, self.w13, self.w2,
            topk_weights, topk_ids, self.num_experts,
        )

        # All-reduce routed output
        if self.tp_size > 1:
            routed_output = self.allreduce(routed_output)

        # Add shared expert (already reduced by RowParallelLinear)
        output = routed_output + shared_output

        return output.view(orig_shape)
