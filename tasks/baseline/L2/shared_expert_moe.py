"""Mixture-of-Experts with optional shared expert (L2).

Consolidates the previously-separate ``KimiMoE`` (Kimi-Linear) and
``Qwen3NextMoE`` (Qwen3-Next) classes into a single configurable module.

Differences between the two are expressed as constructor knobs:

  * ``routing``: ``"sigmoid"`` (Kimi-Linear) or ``"softmax"`` (Qwen3-Next).
  * ``correction_bias``: when True, adds an
    ``e_score_correction_bias`` parameter to the gate; the bias is
    applied *only* to expert selection and the unbiased scores are used
    as routing weights (Kimi-Linear convention).
  * ``shared_expert_intermediate_size``: optional shared SwiGLU MLP added
    after the routed experts. Set to 0 to disable.
  * ``shared_expert_attr_name``: ``"shared_experts"`` (Kimi checkpoint) or
    ``"shared_expert"`` (Qwen3-Next checkpoint) — controls the attribute
    name used for weight loading.
  * ``shared_expert_gate``: when True, an extra replicated linear gates
    the shared expert output via ``sigmoid`` (Qwen3-Next).
  * ``routed_scaling_factor``: multiplier on the routed-expert sum prior
    to all-reduce (Kimi-Linear uses 2.446; Qwen3-Next uses 1.0).
  * ``renormalize``: re-normalize top-k weights to sum to 1.

Checkpoint name compatibility:
  * routed expert weights -> ``self.w13`` / ``self.w2`` (loaded by the
    ``_EXPERT_RE`` / ``_QWEN_NEXT_EXPERT_RE`` paths in ``weight_loader``).
  * router -> ``self.gate.weight`` (and optionally
    ``self.gate.e_score_correction_bias``).
  * shared expert (if any) -> ``self.<shared_expert_attr_name>`` with
    ``gate_up_proj`` / ``down_proj`` children.
  * shared-expert gate (Qwen3-Next) -> ``self.shared_expert_gate.weight``.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.silu_and_mul import SiluAndMul
from ..L2.fused_experts import FusedExperts
from .parallel_linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


class _TPSwiGLUMLP(nn.Module):
    """SwiGLU MLP sharded along TP (gate_up + down projections, SiLU-and-Mul).

    Used both for Kimi-Linear's dense layer-0 MLP and as the optional
    shared expert inside ``SharedExpertMoE``.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size]
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiluAndMul.forward_cuda returns a 2D buffer regardless of input
        # rank, which collapses the leading dims. Save/restore the input
        # shape so downstream code (e.g. KimiLinearDecoderLayer, which passes
        # [B, T, hidden]) sees a same-rank output.
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        gate_up = self.gate_up_proj(x)
        out = self.down_proj(self.act_fn(gate_up))
        return out.view(*orig_shape[:-1], out.shape[-1])


class SharedExpertMoE(nn.Module):
    """Configurable MoE with optional shared expert.

    See module docstring for the per-knob differences between the
    Kimi-Linear and Qwen3-Next configurations.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        moe_intermediate_size: int,
        routing: Literal["sigmoid", "softmax"] = "softmax",
        correction_bias: bool = False,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        shared_expert_intermediate_size: int = 0,
        shared_expert_attr_name: str = "shared_expert",
        shared_expert_gate: bool = False,
    ):
        super().__init__()
        if routing not in ("sigmoid", "softmax"):
            raise ValueError(
                f"routing must be 'sigmoid' or 'softmax', got {routing!r}"
            )

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.routing = routing
        self.correction_bias = correction_bias
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor

        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = moe_intermediate_size // tp

        # Router (replicated, no bias)
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        if correction_bias:
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.zeros(num_experts)
            )
            self.gate.e_score_correction_bias.weight_loader = (
                lambda p, w: p.data.copy_(w)
            )

        # Routed expert weights: w13 = [gate, up] stacked, w2 = down
        N = self.intermediate_per_tp
        self.w13 = nn.Parameter(torch.empty(num_experts, 2 * N, hidden_size))
        self.w13.weight_loader = self._w13_weight_loader
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, N))
        self.w2.weight_loader = self._w2_weight_loader

        self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()

        # Optional shared expert
        self.has_shared_expert = shared_expert_intermediate_size > 0
        self.shared_expert_attr_name = shared_expert_attr_name
        if self.has_shared_expert:
            shared_mlp = _TPSwiGLUMLP(hidden_size, shared_expert_intermediate_size)
            setattr(self, shared_expert_attr_name, shared_mlp)
        if shared_expert_gate:
            if not self.has_shared_expert:
                raise ValueError(
                    "shared_expert_gate=True requires a non-zero "
                    "shared_expert_intermediate_size"
                )
            self.shared_expert_gate = ReplicatedLinear(
                hidden_size, 1, bias=False,
            )
        else:
            self.shared_expert_gate = None

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int,
                           is_w1: bool | None = None,
                           is_gate: bool | None = None):
        """Sharded loader for fused [gate, up] expert weights.

        Accepts either the Kimi (`is_w1`) or Qwen3-Next (`is_gate`) keyword
        argument so the weight_loader callsites in ``infra/weight_loader.py``
        don't need a special case per checkpoint convention.
        """
        if is_w1 is None and is_gate is None:
            raise TypeError("must pass is_w1 or is_gate to w13 loader")
        is_first = bool(is_w1 if is_w1 is not None else is_gate)
        rank = _tp_rank()
        N = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * N, N)
        offset = 0 if is_first else N
        param.data[expert_id, offset:offset + N, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        rank = _tp_rank()
        N = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * N, N))

    def _route(self, router_logits: torch.Tensor):
        """Compute (topk_weights, topk_ids) per the configured routing mode."""
        if self.routing == "sigmoid":
            scores = torch.sigmoid(router_logits.float())
            if self.correction_bias:
                # Bias is added post-sigmoid for SELECTION only; final
                # weights come from the unbiased scores. Matches HF.
                scores_for_choice = scores + self.gate.e_score_correction_bias
                _, topk_ids = scores_for_choice.topk(self.top_k, dim=-1)
                topk_weights = scores.gather(-1, topk_ids)
            else:
                topk_weights, topk_ids = scores.topk(self.top_k, dim=-1)
        else:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = scores.topk(self.top_k, dim=-1)

        if self.renormalize:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        return topk_weights, topk_ids.to(torch.int32)

    def _shared_expert_output(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if not self.has_shared_expert:
            return None
        shared_mlp = getattr(self, self.shared_expert_attr_name)
        out = shared_mlp(hidden_states)
        if self.shared_expert_gate is not None:
            gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            out = out * gate
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)

        shared_output = self._shared_expert_output(hidden_states)

        router_logits = self.gate(hidden_states)
        topk_weights, topk_ids = self._route(router_logits)
        topk_weights = topk_weights.to(hidden_states.dtype)

        routed_output = self.fused_experts(
            hidden_states, self.w13, self.w2,
            topk_weights, topk_ids, self.num_experts,
        )
        if self.routed_scaling_factor != 1.0:
            routed_output = routed_output * self.routed_scaling_factor

        # All-reduce the routed sum first; the shared expert's
        # RowParallelLinear has already all-reduced its own output.
        if self.tp_size > 1:
            routed_output = self.allreduce(routed_output)

        output = routed_output if shared_output is None else routed_output + shared_output
        return output.view(orig_shape)


__all__ = ["SharedExpertMoE", "_TPSwiGLUMLP"]
