"""Kimi-Linear MoE: sigmoid-gated Mixture of Experts with shared expert.

256 routed experts (top-8, sigmoid), 1 shared expert, with
e_score_correction_bias and routed_scaling_factor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L2.fused_experts import FusedExperts
from .parallel_linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)


class KimiMLP(nn.Module):
    """SiLU-gated MLP (shared expert or dense MLP)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size]
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        tp = _tp_size()
        half = gate_up.shape[-1] // 2
        gate, up = gate_up[..., :half], gate_up[..., half:]
        x = torch.nn.functional.silu(gate) * up
        return self.down_proj(x)


class KimiMoE(nn.Module):
    """MoE with sigmoid gating, shared expert, and e_score_correction_bias.

    Checkpoint weight names:
      block_sparse_moe.gate.weight
      block_sparse_moe.gate.e_score_correction_bias
      block_sparse_moe.experts.{j}.w1.weight  (gate_proj)
      block_sparse_moe.experts.{j}.w2.weight  (down_proj)
      block_sparse_moe.experts.{j}.w3.weight  (up_proj)
      block_sparse_moe.shared_experts.gate_proj.weight
      block_sparse_moe.shared_experts.up_proj.weight
      block_sparse_moe.shared_experts.down_proj.weight
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.hidden_size = config.hidden_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.renormalize = config.moe_renormalize
        self.num_shared_experts = config.num_shared_experts
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.moe_intermediate_size // tp

        # Router
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.gate.weight.weight_loader = lambda p, w: p.data.copy_(w)
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts)
        )
        self.gate.e_score_correction_bias.weight_loader = lambda p, w: p.data.copy_(w)

        # Fused expert weights: w13 = [gate, up] stacked, w2 = down
        self.w13 = nn.Parameter(torch.empty(
            config.num_experts,
            2 * self.intermediate_per_tp,
            config.hidden_size,
        ))
        self.w13.weight_loader = self._w13_weight_loader
        self.w2 = nn.Parameter(torch.empty(
            config.num_experts,
            config.hidden_size,
            self.intermediate_per_tp,
        ))
        self.w2.weight_loader = self._w2_weight_loader

        self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()
        self._topk_weights = None
        self._topk_ids = None

        # Shared expert
        if self.num_shared_experts and self.num_shared_experts > 0:
            shared_intermediate = config.moe_intermediate_size * self.num_shared_experts
            self.shared_experts = KimiMLP(config.hidden_size, shared_intermediate)

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

    def _sigmoid_topk(self, router_logits: torch.Tensor):
        """Sigmoid routing with e_score_correction_bias.

        Bias is added post-sigmoid for expert SELECTION only (matching HF).
        Final weights come from unbiased sigmoid scores.
        Returns topk_weights [M, top_k] and topk_ids [M, top_k].
        """
        scores = torch.sigmoid(router_logits.float())
        # Biased scores for selection only
        scores_for_choice = scores + self.gate.e_score_correction_bias
        _, topk_ids = scores_for_choice.topk(self.top_k, dim=-1)
        # Gather weights from UNBIASED scores
        topk_weights = scores.gather(-1, topk_ids)
        if self.renormalize:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_weights.to(router_logits.dtype), topk_ids.to(torch.int32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)

        # Shared expert
        shared_output = None
        if self.num_shared_experts and self.num_shared_experts > 0:
            shared_output = self.shared_experts(hidden_states)

        # Router (bf16, matching vLLM's ReplicatedLinear gate)
        router_logits = torch.nn.functional.linear(
            hidden_states, self.gate.weight
        )
        topk_weights, topk_ids = self._sigmoid_topk(router_logits)
        topk_weights = topk_weights.to(hidden_states.dtype)

        # Routed experts
        routed_output = self.fused_experts(
            hidden_states, self.w13, self.w2,
            topk_weights, topk_ids, self.num_experts,
        )
        final = routed_output * self.routed_scaling_factor

        # All-reduce routed output first (partial sums across TP ranks)
        if self.tp_size > 1:
            final = self.allreduce(final)

        # Add shared expert output AFTER all-reduce (it's already reduced
        # internally by its RowParallelLinear)
        if shared_output is not None:
            final = final + shared_output

        return final.view(orig_shape)
