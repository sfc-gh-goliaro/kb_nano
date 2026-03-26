"""DeepSeek MoE with shared expert, grouped routing, and FP8 expert execution.

Uses GroupedTopK for routing, FusedExperts (BF16) or Fp8MoeGroupedGemm (FP8)
for expert execution, and a shared expert (LlamaMLP) running on a separate stream.

Matches vllm's DeepseekV2MoE: routed_scaling_factor is applied post-experts
(not folded into routing weights), and shared expert uses
moe_intermediate_size * n_shared_experts as its intermediate dimension.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.linear import Linear
from ..L1.silu_and_mul import SiluAndMul
from ..L1.grouped_topk import GroupedTopK
from ..L1.moe_align import MoeAlign
from .fused_experts import FusedExperts
from .parallel_linear import MergedColumnParallelLinear, RowParallelLinear

_FP8_BLOCK = 128


def _scale_shape(out_dim: int, in_dim: int) -> tuple[int, int]:
    return (math.ceil(out_dim / _FP8_BLOCK), math.ceil(in_dim / _FP8_BLOCK))


class DeepSeekSharedExpertMLP(nn.Module):
    """Shared expert MLP with reduce_results=False for external all-reduce."""

    def __init__(self, hidden_size: int, intermediate_size: int,
                 quant_config: dict | None = None):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size,
            quant_config=quant_config,
            reduce_results=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        return self.down_proj(x)


class DeepSeekMoE(nn.Module):
    """DeepSeek Mixture-of-Experts with shared expert and grouped routing.

    Architecture:
    - Router: replicated gate + e_score_correction_bias
    - Shared expert: DeepSeekSharedExpertMLP (reduce_results=False)
    - Routed experts: FP8 weights (w13, w2) + per-block scales
    - Routing: GroupedTopK via sgl_kernel.moe.moe_fused_gate
    - routed_scaling_factor applied post-experts (not in routing weights)
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

        self.gate_weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size),
        )
        self.gate_weight.weight_loader = lambda p, w: p.data.copy_(w)

        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.n_routed_experts, dtype=torch.float32),
        )
        self.e_score_correction_bias.weight_loader = lambda p, w: p.data.copy_(w)

        n_shared = getattr(config, 'n_shared_experts', 1)
        if n_shared is not None and n_shared > 0:
            shared_intermediate = config.moe_intermediate_size * n_shared
            self.shared_expert = DeepSeekSharedExpertMLP(
                config.hidden_size, shared_intermediate,
                quant_config=quant_config,
            )
        else:
            self.shared_expert = None

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

        self.w13.weight_loader = self._w13_weight_loader
        self.w2.weight_loader = self._w2_weight_loader
        if self.use_fp8:
            self.w13_weight_scale_inv.weight_loader = self._w13_scale_loader
            self.w2_weight_scale_inv.weight_loader = self._w2_scale_loader

        self.linear_op = Linear()
        self.grouped_topk = GroupedTopK()
        self.fused_experts = FusedExperts()
        if self.use_fp8:
            self.moe_align = MoeAlign()
        self.silu_and_mul = SiluAndMul()
        self.allreduce = AllReduce()
        self._shared_stream: torch.cuda.Stream | None = None

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

        # Launch shared expert on a separate CUDA stream for overlap
        shared_out = None
        if self.shared_expert is not None:
            if self._shared_stream is None:
                self._shared_stream = torch.cuda.Stream()
            self._shared_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._shared_stream):
                shared_out = self.shared_expert(hidden_states)

        router_logits = self.linear_op(hidden_states, self.gate_weight)
        topk_weights, topk_ids = self.grouped_topk(
            router_logits, self.e_score_correction_bias,
            self.n_group, self.topk_group, self.top_k,
        )
        topk_weights = topk_weights.to(hidden_states.dtype)

        if self.use_fp8:
            out = self._forward_fp8_experts(
                hidden_states, topk_weights, topk_ids)
        else:
            out = self.fused_experts(
                hidden_states, self.w13, self.w2,
                topk_weights, topk_ids, self.num_experts,
            )

        out = out * self.routed_scaling_factor

        if shared_out is not None:
            torch.cuda.current_stream().wait_stream(self._shared_stream)
            out = out + shared_out

        if self.tp_size > 1:
            out = self.allreduce(out)

        return out.view(orig_shape)

    def _forward_fp8_experts(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        from ..L1.fp8_linear import _per_token_group_quant_fp8
        import deep_gemm

        M = hidden_states.shape[0]
        K = self.hidden_size
        N_gate_up = 2 * self.intermediate_per_tp
        N_inter = self.intermediate_per_tp
        top_k = self.top_k

        block_size_m = 128
        sorted_token_ids, expert_ids, num_tokens_post_padded = self.moe_align(
            topk_ids, block_size_m, self.num_experts,
        )

        ntp = int(num_tokens_post_padded.item()
                  if num_tokens_post_padded.numel() == 1
                  else int(num_tokens_post_padded))
        align = int(deep_gemm.get_mk_alignment_for_contiguous_layout())
        m_sum = ((ntp + align - 1) // align) * align

        stp = sorted_token_ids[:ntp].long()
        num_valid = M * top_k
        valid = stp < num_valid
        input_rows = (stp // top_k).clamp(max=M - 1)

        # FP8 quant hidden_states
        num_groups_k = K // _FP8_BLOCK
        a1_fp8 = torch.empty(M, K, dtype=torch.float8_e4m3fn,
                             device=hidden_states.device)
        a1_scale = torch.empty(M, num_groups_k, dtype=torch.float32,
                               device=hidden_states.device)
        _per_token_group_quant_fp8(hidden_states, a1_fp8, a1_scale)

        # Permute for GEMM1
        a1_perm = a1_fp8[input_rows]
        s1_perm = a1_scale[input_rows]
        blk = (torch.arange(ntp, device=hidden_states.device, dtype=torch.int64)
               // block_size_m).clamp(max=expert_ids.shape[0] - 1)
        eid_row = expert_ids[blk].to(torch.int32)
        eid_row = torch.where(valid, eid_row, torch.full_like(eid_row, -1))

        def _pad_to(t, target, fill=0):
            if t.shape[0] >= target:
                return t[:target]
            pad = target - t.shape[0]
            return torch.cat([t, t.new_full((pad, *t.shape[1:]), fill)], dim=0)

        a1_perm = _pad_to(a1_perm, m_sum)
        s1_perm = _pad_to(s1_perm, m_sum)
        eid_row_padded = _pad_to(eid_row, m_sum, fill=-1)

        # GEMM1: gate_up
        out1 = torch.empty(m_sum, N_gate_up, dtype=torch.bfloat16,
                           device=hidden_states.device)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a1_perm, s1_perm), (self.w13, self.w13_weight_scale_inv),
            out1, eid_row_padded,
        )

        # Scatter GEMM1 output to [M*top_k, 2*N]
        gate_up = torch.zeros(M * top_k, N_gate_up, dtype=hidden_states.dtype,
                              device=hidden_states.device)
        out1_valid = out1[:ntp]
        tok_slots = stp.clamp(max=gate_up.shape[0] - 1)
        gate_up[tok_slots[valid]] = out1_valid[valid].to(gate_up.dtype)

        # SiLU + Mul
        intermediate = self.silu_and_mul(gate_up)

        # FP8 quant intermediate for GEMM2 — gather in sorted order
        inter_perm = intermediate[tok_slots]
        inter_perm[~valid] = 0
        num_groups_n = N_inter // _FP8_BLOCK
        a2_fp8 = torch.empty(ntp, N_inter, dtype=torch.float8_e4m3fn,
                             device=hidden_states.device)
        a2_scale = torch.empty(ntp, num_groups_n, dtype=torch.float32,
                               device=hidden_states.device)
        _per_token_group_quant_fp8(inter_perm[:ntp].contiguous(), a2_fp8, a2_scale)

        a2_perm = _pad_to(a2_fp8, m_sum)
        s2_perm = _pad_to(a2_scale, m_sum)

        # GEMM2: down
        out2 = torch.empty(m_sum, K, dtype=torch.bfloat16,
                           device=hidden_states.device)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a2_perm, s2_perm), (self.w2, self.w2_weight_scale_inv),
            out2, eid_row_padded,
        )

        # Scatter GEMM2 output with routed weights to [M*top_k, K]
        down_out = torch.zeros(M * top_k, K, dtype=hidden_states.dtype,
                               device=hidden_states.device)
        out2_valid = out2[:ntp]
        w = topk_weights.view(-1)[tok_slots].unsqueeze(1)
        weighted = (out2_valid * w).to(down_out.dtype)
        weighted[~valid] = 0
        down_out[tok_slots[valid]] = weighted[valid]

        # Sum over top_k
        return down_out.view(M, top_k, K).sum(dim=1)
