"""Mixtral Mixture-of-Experts block with fused Triton grouped GEMM."""

from __future__ import annotations

import torch
import torch.nn as nn

from sgl_kernel.moe import topk_softmax as _sgl_topk_softmax

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.flashinfer_moe import is_available as _fi_available, swap_w13_to_w31
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

        self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()
        self._topk_weights = None
        self._topk_ids = None
        self._topk_weights_bf16 = None
        self._weights_prepared = False

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

    def _prepare_flashinfer_weights(self):
        """Swap W13 -> W31 layout for FlashInfer kernels.

        Called lazily on first forward after weights are loaded.
        """
        if self._weights_prepared:
            return
        self._weights_prepared = True
        if not _fi_available():
            return
        self.w13.data = swap_w13_to_w31(self.w13.data)

    def _ensure_routing_buffers(self, M, device):
        if self._topk_weights is None or self._topk_weights.size(0) < M:
            self._topk_weights = torch.empty(M, self.top_k, device=device, dtype=torch.float32)
            self._topk_ids = torch.empty(M, self.top_k, device=device, dtype=torch.int32)
            self._topk_weights_bf16 = torch.empty(M, self.top_k, device=device, dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self._prepare_flashinfer_weights()

        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        M = hidden_states.size(0)

        router_logits = self.gate(hidden_states)
        self._ensure_routing_buffers(M, hidden_states.device)
        topk_weights = self._topk_weights[:M]
        topk_ids = self._topk_ids[:M]
        _sgl_topk_softmax(topk_weights, topk_ids, router_logits, renormalize=True)
        topk_weights_bf16 = self._topk_weights_bf16[:M]
        topk_weights_bf16.copy_(topk_weights)

        out = self.fused_experts(
            hidden_states, self.w13, self.w2,
            topk_weights_bf16, topk_ids, self.num_experts,
            intermediate_size=self.intermediate_per_tp,
            router_logits=router_logits,
        )

        if self.tp_size > 1:
            out = self.allreduce(out)

        return out.view(orig_shape)
