"""GPT-OSS MoE: softmax-gated MoE with expert biases and OAI SwiGLU.

32 experts (top-4, softmax routing), router bias, expert gate/up/down biases,
OAI SwiGLU activation: (up+1) * gate * sigmoid(alpha*gate) with clamping.

Checkpoint stores gate/up weights INTERLEAVED (gate_0, up_0, gate_1, up_1, ...);
we de-interleave at load time to [gate_all; up_all] for simpler forward pass.

Uses naive expert-loop implementation to support per-expert biases.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sgl_kernel.moe import topk_softmax as _sgl_topk_softmax

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce

# OAI SwiGLU alpha parameter (hardcoded in vLLM's SwigluOAIAndMul)
_SWIGLU_ALPHA = 1.702


class GptOssMoE(nn.Module):
    """MoE with softmax gating, per-expert biases, and OAI SwiGLU.

    Checkpoint weight names (after dequantization):
      layers.{i}.mlp.router.weight
      layers.{i}.mlp.router.bias
      layers.{i}.mlp.experts.w13       (fused gate_up, dequantized from MXFP4)
      layers.{i}.mlp.experts.w2        (down, dequantized from MXFP4)
      layers.{i}.mlp.experts.w13_bias  (gate_up bias)
      layers.{i}.mlp.experts.w2_bias   (down bias)
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.swiglu_limit = config.swiglu_limit
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.intermediate_size // tp

        # Router with bias
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=True)
        self.router.weight.weight_loader = lambda p, w: p.data.copy_(w)
        self.router.bias.weight_loader = lambda p, w: p.data.copy_(w)

        N = self.intermediate_per_tp

        # Expert weights: w13 = [gate, up] stacked, w2 = down
        # Shape: [E, 2*N, K] for w13, [E, K, N] for w2
        self.w13 = nn.Parameter(torch.empty(
            config.num_local_experts, 2 * N, config.hidden_size,
        ))
        self.w13.weight_loader = self._w13_weight_loader

        self.w2 = nn.Parameter(torch.empty(
            config.num_local_experts, config.hidden_size, N,
        ))
        self.w2.weight_loader = self._w2_weight_loader

        # Expert biases
        self.w13_bias = nn.Parameter(torch.zeros(
            config.num_local_experts, 2 * N,
        ))
        self.w13_bias.weight_loader = self._w13_bias_weight_loader

        self.w2_bias = nn.Parameter(torch.zeros(
            config.num_local_experts, config.hidden_size,
        ))
        self.w2_bias.weight_loader = self._w2_bias_weight_loader

        self.allreduce = AllReduce()
        self._topk_weights = None
        self._topk_ids = None

    def _w13_weight_loader(self, param, loaded_weight):
        """Load full dequantized w13 weight [E, 2*N_full, K] -> de-interleave & shard to TP.

        Checkpoint stores gate/up interleaved: [gate_0, up_0, gate_1, up_1, ...]
        We de-interleave to [gate_all; up_all] for simpler forward pass.
        """
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        # De-interleave: even rows = gate, odd rows = up
        gate_all = loaded_weight[:, 0::2, :]  # [E, N_full, K]
        up_all = loaded_weight[:, 1::2, :]    # [E, N_full, K]
        # TP shard
        gate_shard = gate_all[:, rank * N : rank * N + N, :]
        up_shard = up_all[:, rank * N : rank * N + N, :]
        param.data[:, :N, :].copy_(gate_shard)
        param.data[:, N:, :].copy_(up_shard)

    def _w2_weight_loader(self, param, loaded_weight):
        """Load full dequantized w2 weight [E, K, N_full] -> shard to TP."""
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        param.data.copy_(loaded_weight[:, :, rank * N : rank * N + N])

    def _w13_bias_weight_loader(self, param, loaded_weight):
        """Load w13_bias [E, 2*N_full] -> de-interleave & shard to TP.

        Checkpoint stores gate/up bias interleaved: [gate_0, up_0, gate_1, up_1, ...]
        """
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        # De-interleave: even indices = gate, odd indices = up
        gate_all = loaded_weight[:, 0::2]  # [E, N_full]
        up_all = loaded_weight[:, 1::2]    # [E, N_full]
        gate_shard = gate_all[:, rank * N : rank * N + N]
        up_shard = up_all[:, rank * N : rank * N + N]
        param.data[:, :N].copy_(gate_shard)
        param.data[:, N:].copy_(up_shard)

    def _w2_bias_weight_loader(self, param, loaded_weight):
        """Load w2_bias [E, K]. Only rank 0 loads; others zero (reduced by allreduce)."""
        tp, rank = _tp_size(), _tp_rank()
        if rank == 0:
            param.data.copy_(loaded_weight)
        else:
            param.data.zero_()

    def _ensure_routing_buffers(self, M, device):
        if self._topk_weights is None or self._topk_weights.size(0) < M:
            self._topk_weights = torch.empty(M, self.top_k, device=device, dtype=torch.float32)
            self._topk_ids = torch.empty(M, self.top_k, device=device, dtype=torch.int32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        M = hidden_states.size(0)

        # Routing: softmax top-k
        router_logits = self.router(hidden_states)  # [M, E]
        self._ensure_routing_buffers(M, hidden_states.device)
        topk_weights = self._topk_weights[:M]
        topk_ids = self._topk_ids[:M]
        _sgl_topk_softmax(topk_weights, topk_ids, router_logits, renormalize=True)
        topk_weights = topk_weights.to(hidden_states.dtype)

        # Naive expert loop with biases and clamped SwiGLU
        output = torch.zeros(M, self.hidden_size, device=hidden_states.device,
                             dtype=hidden_states.dtype)

        for e in range(self.num_experts):
            # Find all (token, topk_slot) assignments to this expert
            mask = (topk_ids == e)  # [M, top_k]
            if not mask.any():
                continue

            # Get unique token indices
            token_mask = mask.any(dim=1)  # [M]
            x = hidden_states[token_mask]  # [n, K]

            # Expert forward: gate_up with bias
            gate_up = x @ self.w13[e].T + self.w13_bias[e]  # [n, 2N]
            N = gate_up.shape[-1] // 2
            gate, up = gate_up[..., :N], gate_up[..., N:]

            # OAI SwiGLU activation: (up + 1) * gate * sigmoid(alpha * gate)
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            glu = gate * torch.sigmoid(gate * _SWIGLU_ALPHA)
            h = (up + 1) * glu

            # Down projection with bias
            out = h @ self.w2[e].T + self.w2_bias[e]  # [n, K]

            # Combined weight: sum of topk_weights where this expert is assigned
            combined_weight = (topk_weights[token_mask] * mask[token_mask].float()).sum(
                dim=1, keepdim=True,
            )  # [n, 1]
            output[token_mask] += combined_weight.to(out.dtype) * out

        if self.tp_size > 1:
            output = self.allreduce(output)

        return output.view(orig_shape)
