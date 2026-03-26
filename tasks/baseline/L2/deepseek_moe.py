"""DeepSeek MoE with shared expert, grouped routing, and FP8 expert execution.

Uses GroupedTopK for routing, FusedExperts (BF16) or Fp8MoeGroupedGemm (FP8)
for expert execution, and a shared expert (LlamaMLP) running on a separate stream.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.linear import Linear
from ..L1.grouped_topk import GroupedTopK
from .fused_experts import FusedExperts
from .llama_mlp import LlamaMLP

_FP8_BLOCK = 128


def _scale_shape(out_dim: int, in_dim: int) -> tuple[int, int]:
    return (math.ceil(out_dim / _FP8_BLOCK), math.ceil(in_dim / _FP8_BLOCK))


class DeepSeekMoE(nn.Module):
    """DeepSeek Mixture-of-Experts with shared expert and grouped routing.

    Architecture:
    - Router: replicated gate + e_score_correction_bias
    - Shared expert: LlamaMLP (reused, FP8 via quant_config)
    - Routed experts: FP8 weights (w13, w2) + per-block scales
    - Routing: GroupedTopK via sgl_kernel.moe.moe_fused_gate
    """

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.moe_intermediate_size // tp
        self.use_fp8 = quant_config is not None

        n_group = getattr(config, 'n_group', 1)
        topk_group = getattr(config, 'topk_group', 1)
        self.n_group = n_group
        self.topk_group = topk_group

        # Router gate (replicated, no TP, no FP8)
        self.gate_weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size),
        )
        self.gate_weight.weight_loader = lambda p, w: p.data.copy_(w)

        # Correction bias for noaux_tc routing
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.n_routed_experts, dtype=torch.float32),
        )
        self.e_score_correction_bias.weight_loader = lambda p, w: p.data.copy_(w)

        # Shared expert
        n_shared = getattr(config, 'n_shared_experts', 1)
        if n_shared is not None and n_shared > 0:
            self.shared_expert = LlamaMLP(config, quant_config=quant_config)
        else:
            self.shared_expert = None

        # Expert weights
        if self.use_fp8:
            self.w13 = nn.Parameter(torch.empty(
                config.n_routed_experts, 2 * self.intermediate_per_tp, config.hidden_size,
                dtype=torch.float8_e4m3fn,
            ), requires_grad=False)
            self.w2 = nn.Parameter(torch.empty(
                config.n_routed_experts, config.hidden_size, self.intermediate_per_tp,
                dtype=torch.float8_e4m3fn,
            ), requires_grad=False)
            self.w13_weight_scale_inv = nn.Parameter(torch.empty(
                config.n_routed_experts,
                *_scale_shape(2 * self.intermediate_per_tp, config.hidden_size),
                dtype=torch.float32,
            ), requires_grad=False)
            self.w2_weight_scale_inv = nn.Parameter(torch.empty(
                config.n_routed_experts,
                *_scale_shape(config.hidden_size, self.intermediate_per_tp),
                dtype=torch.float32,
            ), requires_grad=False)
        else:
            self.w13 = nn.Parameter(torch.empty(
                config.n_routed_experts, 2 * self.intermediate_per_tp, config.hidden_size,
            ))
            self.w2 = nn.Parameter(torch.empty(
                config.n_routed_experts, config.hidden_size, self.intermediate_per_tp,
            ))

        # Weight loaders
        self.w13.weight_loader = self._w13_weight_loader
        self.w2.weight_loader = self._w2_weight_loader
        if self.use_fp8:
            self.w13_weight_scale_inv.weight_loader = self._w13_scale_loader
            self.w2_weight_scale_inv.weight_loader = self._w2_scale_loader

        self.linear_op = Linear()
        self.grouped_topk = GroupedTopK()
        self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()

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

    def _w13_scale_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        scale_rows = math.ceil(N / _FP8_BLOCK)
        offset = 0 if is_w1 else scale_rows
        src = loaded_weight.chunk(tp, 0)[rank]
        param.data[expert_id, offset:offset + scale_rows, :].copy_(src)

    def _w2_scale_loader(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        scale_cols = math.ceil(N / _FP8_BLOCK)
        src = loaded_weight.chunk(tp, 1)[rank]
        param.data[expert_id].copy_(src)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        # Router
        router_logits = self.linear_op(hidden_states, self.gate_weight)
        topk_weights, topk_ids = self.grouped_topk(
            router_logits, self.e_score_correction_bias,
            self.n_group, self.topk_group, self.top_k,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        topk_weights = topk_weights.to(hidden_states.dtype)

        # Routed experts (BF16 path for now, FP8 TODO)
        out = self.fused_experts(
            hidden_states, self.w13 if not self.use_fp8 else self.w13.float().to(hidden_states.dtype),
            self.w2 if not self.use_fp8 else self.w2.float().to(hidden_states.dtype),
            topk_weights, topk_ids, self.num_experts,
        )

        # Shared expert
        if self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states)
            out = out + shared_out

        if self.tp_size > 1:
            out = self.allreduce(out)

        return out.view(orig_shape)
