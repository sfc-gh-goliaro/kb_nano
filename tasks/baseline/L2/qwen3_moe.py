"""Qwen3 Mixture-of-Experts block with FP8 W8A8 block-scaled grouped GEMM.

128 experts, top-8 routing, moe_intermediate_size=768.
Expert weights stored as fused 3D FP8 tensors with per-block scale factors.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sgl_kernel.moe import topk_softmax as _sgl_topk_softmax

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.flashinfer_moe import is_available as _fi_available, swap_w13_to_w31
from .fused_experts import FusedExperts


def _ceildiv(a: int, b: int) -> int:
    return -(-a // b)


class Qwen3MoE(nn.Module):
    """Qwen3 MoE with FP8 expert weights.

    Weights:
      w13: [E, 2*moe_intermediate_per_tp, hidden_size] float8_e4m3fn
      w13_scale_inv: [E, ceil(2*moe_intermediate_per_tp/bn), ceil(hidden_size/bk)] float32
      w2:  [E, hidden_size, moe_intermediate_per_tp] float8_e4m3fn
      w2_scale_inv: [E, ceil(hidden_size/bn), ceil(moe_intermediate_per_tp/bk)] float32
      gate: [num_experts, hidden_size] BF16
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)
        tp = _tp_size()
        self.tp_size = tp
        self.moe_intermediate_per_tp = config.moe_intermediate_size // tp

        fp8_block_size = getattr(config, "fp8_block_size", None)
        assert fp8_block_size is not None, "Qwen3MoE requires FP8 block size"
        self._fp8_block_size = fp8_block_size
        bs_n, bs_k = fp8_block_size

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.gate.weight.weight_loader = lambda p, w: p.data.copy_(w)

        w13_out = 2 * self.moe_intermediate_per_tp
        self.w13 = nn.Parameter(torch.empty(
            config.num_experts, w13_out, config.hidden_size,
            dtype=torch.float8_e4m3fn,
        ), requires_grad=False)
        self.w13.weight_loader = self._w13_weight_loader

        self.w13_scale_inv = nn.Parameter(torch.ones(
            config.num_experts,
            _ceildiv(w13_out, bs_n),
            _ceildiv(config.hidden_size, bs_k),
            dtype=torch.float32,
        ), requires_grad=False)
        self.w13_scale_inv.weight_loader = self._w13_scale_weight_loader

        self.w2 = nn.Parameter(torch.empty(
            config.num_experts, config.hidden_size, self.moe_intermediate_per_tp,
            dtype=torch.float8_e4m3fn,
        ), requires_grad=False)
        self.w2.weight_loader = self._w2_weight_loader

        self.w2_scale_inv = nn.Parameter(torch.ones(
            config.num_experts,
            _ceildiv(config.hidden_size, bs_n),
            _ceildiv(self.moe_intermediate_per_tp, bs_k),
            dtype=torch.float32,
        ), requires_grad=False)
        self.w2_scale_inv.weight_loader = self._w2_scale_weight_loader

        self.fused_experts = FusedExperts(block_size=fp8_block_size)
        self.allreduce = AllReduce()
        self._topk_weights = None
        self._topk_ids = None
        self._topk_weights_bf16 = None
        self._weights_prepared = False

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int = -1,
                           is_w1: bool = True):
        """Load fused gate_up_proj expert weights.

        Checkpoint stores gate_up_proj as [E, hidden, 2*intermediate].
        Our param is [E, 2*intermediate_per_tp, hidden] so we transpose
        dims 1,2 then split and shard the intermediate dimension.
        """
        tp, rank = _tp_size(), _tp_rank()
        N = self.moe_intermediate_per_tp
        if expert_id == -1:
            # Checkpoint: [E, hidden, 2*intermediate] -> transpose to [E, 2*intermediate, hidden]
            w = loaded_weight.transpose(1, 2)
            gate = w[:, :w.shape[1] // 2, :]
            up = w[:, w.shape[1] // 2:, :]
            gate_shard = gate[:, rank * N:(rank + 1) * N, :]
            up_shard = up[:, rank * N:(rank + 1) * N, :]
            param.data[:, :N, :].copy_(gate_shard)
            param.data[:, N:, :].copy_(up_shard)
        else:
            # Per-expert: [hidden, intermediate] -> [intermediate, hidden]
            w = loaded_weight.t()
            shard = w.narrow(0, rank * N, N)
            offset = 0 if is_w1 else N
            param.data[expert_id, offset:offset + N, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int = -1):
        """Load down_proj expert weights.

        Checkpoint stores w2 as [E, intermediate, hidden]. Our param is
        [E, hidden, intermediate_per_tp] so we transpose dims 1,2 then
        shard the (now last) intermediate dimension.
        """
        tp, rank = _tp_size(), _tp_rank()
        N = self.moe_intermediate_per_tp
        if expert_id == -1:
            # loaded: [E, intermediate, hidden] -> transpose to [E, hidden, intermediate]
            w = loaded_weight.transpose(1, 2)
            param.data.copy_(w[:, :, rank * N:(rank + 1) * N])
        else:
            # loaded: [intermediate, hidden] -> [hidden, intermediate]
            w = loaded_weight.t()
            param.data[expert_id].copy_(w[:, rank * N:(rank + 1) * N])

    def _w13_scale_weight_loader(self, param, loaded_weight):
        """Load fused gate_up_proj scale tensor.

        Checkpoint stores scales for gate_up_proj=[E, hidden, 2*intermediate]
        layout as [E, ceil(hidden/bn), ceil(2*intermediate/bk)].
        Our param matches [E, 2*intermediate_per_tp, hidden] layout so we
        transpose to [E, ceil(2*intermediate/bk), ceil(hidden/bn)] then
        split and shard.
        """
        tp, rank = _tp_size(), _tp_rank()
        bs_n = self._fp8_block_size[0]
        N = self.moe_intermediate_per_tp
        scale_rows_per_shard = _ceildiv(N, bs_n)
        # Transpose scale dims to match transposed weight layout
        s = loaded_weight.transpose(1, 2)
        gate_scales = s[:, :s.shape[1] // 2, :]
        up_scales = s[:, s.shape[1] // 2:, :]
        gate_shard = gate_scales[:, rank * scale_rows_per_shard:(rank + 1) * scale_rows_per_shard, :]
        up_shard = up_scales[:, rank * scale_rows_per_shard:(rank + 1) * scale_rows_per_shard, :]
        param.data[:, :scale_rows_per_shard, :].copy_(gate_shard)
        param.data[:, scale_rows_per_shard:, :].copy_(up_shard)

    def _w2_scale_weight_loader(self, param, loaded_weight):
        """Load down_proj scale tensor.

        Checkpoint stores scales for w2=[E, intermediate, hidden] layout as
        [E, ceil(intermediate/bn), ceil(hidden/bk)].
        Our param matches w2=[E, hidden, intermediate_per_tp] layout so we
        transpose to [E, ceil(hidden/bn), ceil(intermediate/bk)] then shard.
        """
        tp, rank = _tp_size(), _tp_rank()
        bs_k = self._fp8_block_size[1]
        N = self.moe_intermediate_per_tp
        scale_cols_per_shard = _ceildiv(N, bs_k)
        # Transpose scale dims to match transposed weight layout
        s = loaded_weight.transpose(1, 2)
        param.data.copy_(s[:, :, rank * scale_cols_per_shard:(rank + 1) * scale_cols_per_shard])

    def _prepare_flashinfer_weights(self):
        """Swap W13 -> W31 layout and clamp scales for FlashInfer kernels.

        Called lazily on first forward after weights are loaded.
        """
        if self._weights_prepared:
            return
        self._weights_prepared = True
        if not _fi_available():
            return
        self.w13.data = swap_w13_to_w31(self.w13.data)
        self.w13_scale_inv.data = swap_w13_to_w31(self.w13_scale_inv.data)
        _MIN_BLOCK_SCALE = 1e-10
        self.w13_scale_inv.data.clamp_(min=_MIN_BLOCK_SCALE)
        self.w2_scale_inv.data.clamp_(min=_MIN_BLOCK_SCALE)

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
        _sgl_topk_softmax(topk_weights, topk_ids, router_logits,
                          renormalize=self.norm_topk_prob)
        topk_weights_bf16 = self._topk_weights_bf16[:M]
        topk_weights_bf16.copy_(topk_weights)

        out = self.fused_experts(
            hidden_states, self.w13, self.w2,
            topk_weights_bf16, topk_ids, self.num_experts,
            w13_scale_inv=self.w13_scale_inv,
            w2_scale_inv=self.w2_scale_inv,
            intermediate_size=self.moe_intermediate_per_tp,
        )

        if self.tp_size > 1:
            out = self.allreduce(out)

        return out.view(orig_shape)
