"""Llama 4 MoE: sigmoid top-1 routing with shared expert.

Router uses sigmoid (not softmax). Output = routed_expert + shared_expert.

Weight names match checkpoint:
  feed_forward.router.weight                           [num_experts, hidden_size]
  feed_forward.shared_expert.gate_up_proj.weight       [2*intermediate, hidden_size]  (packed)
  feed_forward.shared_expert.down_proj.weight          [hidden_size, intermediate]
  feed_forward.w13                                     [E, 2*inter_per_tp, hidden_size]
  feed_forward.w2                                      [E, hidden_size, inter_per_tp]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L2.llama_mlp import LlamaMLP
from ..L2.fused_experts import FusedExperts


class _SharedExpertConfig:
    """Minimal config object for LlamaMLP."""
    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size


class Llama4MoE(nn.Module):
    """MoE with sigmoid top-1 routing and shared expert.

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

        # Router: replicated across TP ranks
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.router.weight.weight_loader = lambda p, w: p.data.copy_(w)

        # Shared expert: standard SwiGLU MLP (skip its internal all-reduce;
        # we do a single all-reduce on shared + routed together)
        shared_cfg = _SharedExpertConfig(config.hidden_size, config.intermediate_size)
        self.shared_expert = LlamaMLP(shared_cfg)
        if tp > 1:
            self.shared_expert.down_proj.tp_size = 1

        # Routed experts: fused [E, 2*N, K] and [E, K, N]
        self.w13 = nn.Parameter(torch.empty(
            config.num_local_experts, 2 * self.intermediate_per_tp, config.hidden_size,
        ))
        self.w13.weight_loader = self._w13_weight_loader

        self.w2 = nn.Parameter(torch.empty(
            config.num_local_experts, config.hidden_size, self.intermediate_per_tp,
        ))
        self.w2.weight_loader = self._w2_weight_loader

        self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()
        self._topk_weights = None
        self._topk_ids = None

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int = 0,
                           is_gate: bool = True):
        """Load a single expert's gate or up weight into the fused w13."""
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * N, N)
        offset = 0 if is_gate else N
        param.data[expert_id, offset:offset + N, :].copy_(shard)

    def _w13_fused_weight_loader(self, param, loaded_weight):
        """Load all experts' fused gate_up_proj: [E, in, 2*out] → transpose → w13."""
        tp, rank = _tp_size(), _tp_rank()
        # loaded_weight: [E, hidden, 2*intermediate]
        weight = loaded_weight.transpose(-1, -2)  # [E, 2*intermediate, hidden]
        N = self.intermediate_per_tp
        # Shard gate and up separately
        gate = weight[:, :weight.shape[1] // 2, :]  # [E, intermediate, hidden]
        up = weight[:, weight.shape[1] // 2:, :]    # [E, intermediate, hidden]
        param.data[:, :N, :].copy_(gate[:, rank * N:(rank + 1) * N, :])
        param.data[:, N:, :].copy_(up[:, rank * N:(rank + 1) * N, :])

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int = 0):
        """Load a single expert's down weight into w2."""
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * N, N))

    def _w2_fused_weight_loader(self, param, loaded_weight):
        """Load all experts' fused down_proj: [E, out, in] → transpose → w2."""
        tp, rank = _tp_size(), _tp_rank()
        # loaded_weight: [E, intermediate, hidden]
        weight = loaded_weight.transpose(-1, -2)  # [E, hidden, intermediate]
        N = self.intermediate_per_tp
        param.data.copy_(weight[:, :, rank * N:(rank + 1) * N])

    def _ensure_routing_buffers(self, M, device):
        if self._topk_weights is None or self._topk_weights.size(0) < M:
            self._topk_weights = torch.empty(M, self.top_k, device=device, dtype=torch.float32)
            self._topk_ids = torch.empty(M, self.top_k, device=device, dtype=torch.int32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        M = hidden_states.size(0)

        # Shared expert
        shared_out = self.shared_expert(hidden_states)

        # Sigmoid top-k routing
        router_logits = self.router(hidden_states)  # [M, E]
        self._ensure_routing_buffers(M, hidden_states.device)
        topk_weights = self._topk_weights[:M]
        topk_ids = self._topk_ids[:M]
        scores, indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights.copy_(torch.sigmoid(scores.float()))
        topk_ids.copy_(indices.to(torch.int32))
        # Apply routing weight on input (matches vLLM's apply_router_weight_on_input=True).
        # SiLU is nonlinear so w*expert(x) != expert(w*x); Llama4 uses the latter.
        weighted_input = hidden_states * topk_weights.to(hidden_states.dtype)

        # Routed experts with unit weights (partial per-rank result, needs all-reduce)
        routed_out = self.fused_experts(
            weighted_input, self.w13, self.w2,
            self._ones_weights(M, hidden_states), topk_ids, self.num_experts,
        )

        # Single all-reduce on the sum (matches vLLM's reduce_results=False
        # on shared_expert + single all-reduce at the end).
        out = routed_out + shared_out
        if self.tp_size > 1:
            out = self.allreduce(out)
        return out.view(orig_shape)

    def _ones_weights(self, M, ref_tensor):
        if not hasattr(self, '_ones_buf') or self._ones_buf.size(0) < M:
            self._ones_buf = torch.ones(
                M, self.top_k, device=ref_tensor.device, dtype=ref_tensor.dtype,
            )
        return self._ones_buf[:M]
