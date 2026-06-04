"""DeepSeek V4 MoE with hash routing, sqrtsoftplus scoring, and MXFP4 experts.

Key differences from V3 MoE:
- sqrtsoftplus scoring function (instead of sigmoid)
- Hash MoE routing for first num_hash_layers layers (vocab lookup table)
- SiluAndMulWithClamp activation (swiglu_limit)
- MXFP4 expert weights (dequantized to BF16 for computation)
- Different expert count and routing factor

Expert weights are stored as packed int8 (2 FP4 values per byte) with
E8M0 scales. During loading, they are dequantized to BF16.

Reference: vllm/model_executor/models/deepseek_v4.py:DeepseekV4MoE
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.gate_linear import GateLinear
from .llama_mlp import LlamaMLP
from .fused_experts import FusedExperts


def sqrtsoftplus(x: torch.Tensor) -> torch.Tensor:
    """sqrtsoftplus scoring: sqrt(softplus(x)) = sqrt(log(1 + exp(x)))"""
    return torch.sqrt(F.softplus(x))


def _dequant_mxfp4(packed_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 packed int8 + E8M0 scale to BF16.

    packed_weight: (..., N) int8 where each byte holds 2 FP4 values
    scale: (..., N/16) uint8 E8M0 exponent-only scales (1 scale per 32 FP4 values = 16 bytes)
    """
    _FP4_E2M1_LUT = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ], dtype=torch.float32)

    device = packed_weight.device
    _FP4_E2M1_LUT = _FP4_E2M1_LUT.to(device)

    shape = packed_weight.shape
    flat = packed_weight.view(-1).to(torch.uint8)

    low = (flat & 0x0F).long()
    high = ((flat >> 4) & 0x0F).long()
    unpacked = torch.stack([low, high], dim=-1).reshape(-1)
    values = _FP4_E2M1_LUT[unpacked]

    # Reshape to match weight layout
    new_shape = list(shape[:-1]) + [shape[-1] * 2]
    values = values.view(new_shape)

    # Apply E8M0 scales: each scale covers 32 consecutive elements
    scale_uint8 = scale.view(torch.uint8) if scale.dtype != torch.uint8 else scale
    exponents = scale_uint8.to(torch.int32) - 127
    scale_values = (2.0 ** exponents.float())

    # Expand scales to match weight dimensions
    # scale: (..., N/16), values: (..., N*2) where N*2 = (N/16)*32
    scale_expanded = scale_values.unsqueeze(-1).expand(
        *scale_values.shape, 32
    ).reshape(*scale_values.shape[:-1], -1)

    result = values * scale_expanded
    return result.to(torch.bfloat16)


def _fused_topk_bias(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    e_score_correction_bias: torch.Tensor | None,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    input_ids: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route tokens to experts using sqrtsoftplus scoring."""
    if hash_indices_table is not None and input_ids is not None:
        topk_ids = hash_indices_table[input_ids.long()]
        scores = sqrtsoftplus(gating_output)
        topk_weights = scores.gather(1, topk_ids.long())
        if renormalize:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * routed_scaling_factor
        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    scores = sqrtsoftplus(gating_output)

    if e_score_correction_bias is not None:
        scores_with_bias = scores + e_score_correction_bias.unsqueeze(0)
        _, topk_ids = torch.topk(scores_with_bias, k=topk, dim=-1)
    else:
        _, topk_ids = torch.topk(scores, k=topk, dim=-1)

    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


class DeepSeekV4MoE(nn.Module):
    """DeepSeek V4 Mixture-of-Experts.

    Expert weights are loaded from MXFP4 checkpoint and dequantized to BF16.
    Shared expert uses FP8 from the existing LlamaMLP infrastructure.
    """

    def __init__(self, config, layer_idx: int, quant_config: dict | None = None):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.5)
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_size = config.moe_intermediate_size
        self.intermediate_per_tp = config.moe_intermediate_size // tp
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)
        self.swiglu_limit = getattr(config, 'swiglu_limit', 10.0)

        self.is_hash_moe = layer_idx < getattr(config, 'num_hash_layers', 3)

        # Gate (router)
        self.gate_weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size),
        )
        self.gate_weight.weight_loader = lambda p, w: p.data.copy_(w)

        if self.is_hash_moe:
            self.tid2eid = nn.Parameter(
                torch.randint(
                    0, config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            self.tid2eid.weight_loader = lambda p, w: p.data.copy_(w)
            self.register_parameter('e_score_correction_bias', None)
        else:
            self.register_parameter('tid2eid', None)
            if getattr(config, 'topk_method', None) == 'noaux_tc':
                self.e_score_correction_bias = nn.Parameter(
                    torch.zeros(config.n_routed_experts, dtype=torch.float32),
                )
                self.e_score_correction_bias.weight_loader = (
                    lambda p, w: p.data.copy_(w)
                )
            else:
                self.register_parameter('e_score_correction_bias', None)

        # Shared expert (FP8 via LlamaMLP)
        n_shared = getattr(config, 'n_shared_experts', 1)
        if n_shared is not None and n_shared > 0:
            shared_intermediate = config.moe_intermediate_size * n_shared
            self.shared_expert = LlamaMLP(
                config,
                quant_config=quant_config,
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate,
                reduce_results=False,
                swiglu_limit=self.swiglu_limit,
            )
        else:
            self.shared_expert = None

        # Expert weights: BF16 (dequantized from MXFP4 during loading)
        self.w13 = nn.Parameter(torch.empty(
            config.n_routed_experts, 2 * self.intermediate_per_tp, config.hidden_size,
            dtype=torch.bfloat16,
        ), requires_grad=False)
        self.w2 = nn.Parameter(torch.empty(
            config.n_routed_experts, config.hidden_size, self.intermediate_per_tp,
            dtype=torch.bfloat16,
        ), requires_grad=False)

        self.w13.weight_loader = self._w13_weight_loader
        self.w2.weight_loader = self._w2_weight_loader

        # Stores for deferred MXFP4 dequantization
        self._w13_mxfp4_pending: dict = {}
        self._w2_mxfp4_pending: dict = {}

        self.gate = GateLinear()
        self.fused_experts = FusedExperts(swiglu_limit=self.swiglu_limit)
        self.allreduce = AllReduce()

        self._shared_stream: torch.cuda.Stream | None = None
        import os as _os
        self._disable_shared_stream: bool = (
            _os.environ.get("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0") != "0"
        )

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        """Load w1 or w3 expert weight.

        If MXFP4 (int8 packed), store for deferred dequantization.
        If BF16/FP8, load directly.
        """
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        offset = 0 if is_w1 else N

        if loaded_weight.dtype == torch.int8 or loaded_weight.dtype == torch.uint8:
            key = (expert_id, is_w1)
            if key not in self._w13_mxfp4_pending:
                self._w13_mxfp4_pending[key] = {}
            self._w13_mxfp4_pending[key]['weight'] = loaded_weight
            self._try_dequant_w13(key)
        else:
            shard = loaded_weight.narrow(0, rank * N, N)
            param.data[expert_id, offset:offset + N, :].copy_(shard.to(param.dtype))

    def _w13_scale_loader(self, param_unused, loaded_weight, expert_id: int, is_w1: bool):
        """Load MXFP4 scale for w1/w3 expert."""
        key = (expert_id, is_w1)
        if key not in self._w13_mxfp4_pending:
            self._w13_mxfp4_pending[key] = {}
        if loaded_weight.dtype == torch.float8_e8m0fnu:
            loaded_weight = loaded_weight.view(torch.uint8)
        self._w13_mxfp4_pending[key]['scale'] = loaded_weight
        self._try_dequant_w13(key)

    def _try_dequant_w13(self, key):
        """Dequantize w13 MXFP4 weight once both weight and scale are available."""
        pending = self._w13_mxfp4_pending.get(key, {})
        if 'weight' not in pending or 'scale' not in pending:
            return
        expert_id, is_w1 = key
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        offset = 0 if is_w1 else N

        bf16_weight = _dequant_mxfp4(pending['weight'], pending['scale'])
        shard = bf16_weight.narrow(0, rank * N, N)
        self.w13.data[expert_id, offset:offset + N, :].copy_(shard)
        del self._w13_mxfp4_pending[key]

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        """Load w2 expert weight."""
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp

        if loaded_weight.dtype == torch.int8 or loaded_weight.dtype == torch.uint8:
            if expert_id not in self._w2_mxfp4_pending:
                self._w2_mxfp4_pending[expert_id] = {}
            self._w2_mxfp4_pending[expert_id]['weight'] = loaded_weight
            self._try_dequant_w2(expert_id)
        else:
            param.data[expert_id].copy_(
                loaded_weight.narrow(1, rank * N, N).to(param.dtype)
            )

    def _w2_scale_loader(self, param_unused, loaded_weight, expert_id: int):
        """Load MXFP4 scale for w2 expert."""
        if expert_id not in self._w2_mxfp4_pending:
            self._w2_mxfp4_pending[expert_id] = {}
        if loaded_weight.dtype == torch.float8_e8m0fnu:
            loaded_weight = loaded_weight.view(torch.uint8)
        self._w2_mxfp4_pending[expert_id]['scale'] = loaded_weight
        self._try_dequant_w2(expert_id)

    def _try_dequant_w2(self, expert_id):
        """Dequantize w2 MXFP4 weight once both weight and scale are available."""
        pending = self._w2_mxfp4_pending.get(expert_id, {})
        if 'weight' not in pending or 'scale' not in pending:
            return
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp

        bf16_weight = _dequant_mxfp4(pending['weight'], pending['scale'])
        self.w2.data[expert_id].copy_(
            bf16_weight.narrow(1, rank * N, N)
        )
        del self._w2_mxfp4_pending[expert_id]

    def forward(self, hidden_states: torch.Tensor,
                input_ids: torch.Tensor | None = None) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        # Shared expert on separate stream
        shared_out = None
        use_shared_stream = (
            self.shared_expert is not None and not self._disable_shared_stream
        )
        if use_shared_stream:
            if self._shared_stream is None:
                self._shared_stream = torch.cuda.Stream()
            self._shared_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._shared_stream):
                shared_out = self.shared_expert(hidden_states)
        elif self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states)

        # Router
        router_logits = self.gate(
            hidden_states, self.gate_weight, out_dtype=torch.float32,
        )

        topk_weights, topk_ids = _fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            e_score_correction_bias=self.e_score_correction_bias,
            topk=self.top_k,
            renormalize=self.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
            input_ids=input_ids,
            hash_indices_table=self.tid2eid,
        )

        # Expert execution (BF16)
        topk_weights_bf16 = topk_weights.to(hidden_states.dtype)
        out = self.fused_experts(
            hidden_states, self.w13, self.w2,
            topk_weights_bf16, topk_ids, self.num_experts,
        )

        if shared_out is not None:
            if use_shared_stream:
                torch.cuda.current_stream().wait_stream(self._shared_stream)
            out = out + shared_out

        if self.tp_size > 1:
            out = self.allreduce(out)

        return out.view(orig_shape)
