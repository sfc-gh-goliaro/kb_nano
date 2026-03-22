"""DeepSeek V3 MoE: grouped sigmoid top-k routing with shared expert.

Supports expert parallelism (EP) via all_gather dispatch and
reduce_scatter combine across EP ranks.

Weight layout:
  gate.weight:                  [n_routed_experts, hidden_size]  (replicated)
  e_score_correction_bias:      [n_routed_experts]               (replicated, noaux_tc)
  shared_experts.gate_up_proj:  [2*moe_intermediate, hidden_size]
  shared_experts.down_proj:     [hidden_size, moe_intermediate]
  experts.w13:                  [n_local_experts, 2*moe_intermediate, hidden_size]
  experts.w2:                   [n_local_experts, hidden_size, moe_intermediate]
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn

from ....infra.tp import _tp_rank, _tp_size, _ep_size, _ep_rank, get_ep_group
from ..L1.allreduce import AllReduce
from ..L1.sigmoid_topk import GroupedSigmoidTopK
from ..L2.llama_mlp import LlamaMLP
from ..L2.fused_experts import FusedExperts

_MOE_DP_CHUNK_SIZE = 256

_ep_cached_max_n: int | None = None

def set_ep_max_n(max_n: int | None) -> None:
    global _ep_cached_max_n
    _ep_cached_max_n = max_n


class _SharedExpertConfig:
    """Minimal config object for LlamaMLP."""
    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size


class DeepSeekMoE(nn.Module):
    """DeepSeek V3 MoE with grouped sigmoid routing and EP support.

    Each EP rank holds n_routed_experts // ep_size local experts.
    The router is replicated. Dispatch gathers all tokens via all_gather,
    runs local experts, then reduce_scatter sends results back.
    """

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)

        ep = _ep_size()
        self.ep_size = ep
        self.ep_rank = _ep_rank()
        self.n_local_experts = self.n_routed_experts // ep
        self.expert_start = self.ep_rank * self.n_local_experts
        self.expert_end = self.expert_start + self.n_local_experts

        tp = _tp_size()
        self.tp_size = tp

        moe_intermediate = config.moe_intermediate_size

        # Router gate (replicated, matches checkpoint key: mlp.gate.weight)
        self.gate = nn.Linear(self.hidden_size, self.n_routed_experts, bias=False)
        self.gate.weight.weight_loader = lambda p, w: p.data.copy_(w)

        # noaux_tc correction bias
        topk_method = getattr(config, "topk_method", "")
        if topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.zeros(self.n_routed_experts),
            )
            self.e_score_correction_bias.weight_loader = lambda p, w: p.data.copy_(w)
        else:
            self.e_score_correction_bias = None

        # Shared expert (full MLP, no TP sharding on its own)
        if self.n_shared_experts and self.n_shared_experts > 0:
            shared_cfg = _SharedExpertConfig(
                self.hidden_size,
                moe_intermediate * self.n_shared_experts,
            )
            self.shared_experts = LlamaMLP(shared_cfg, quant_config=quant_config)
            if tp > 1:
                self.shared_experts.down_proj.tp_size = 1
        else:
            self.shared_experts = None

        # Local expert weights (FP8 when quantized to save memory)
        expert_dtype = torch.float8_e4m3fn if quant_config is not None else None
        if expert_dtype is not None:
            self.w13 = nn.Parameter(torch.empty(
                self.n_local_experts, 2 * moe_intermediate, self.hidden_size,
                dtype=expert_dtype,
            ), requires_grad=False)
            self.w2 = nn.Parameter(torch.empty(
                self.n_local_experts, self.hidden_size, moe_intermediate,
                dtype=expert_dtype,
            ), requires_grad=False)
        else:
            self.w13 = nn.Parameter(torch.empty(
                self.n_local_experts, 2 * moe_intermediate, self.hidden_size,
            ))
            self.w2 = nn.Parameter(torch.empty(
                self.n_local_experts, self.hidden_size, moe_intermediate,
            ))
        self.w13.weight_loader = self._w13_weight_loader
        self.w2.weight_loader = self._w2_weight_loader

        # Block scales for FP8 expert weights
        if quant_config is not None:
            import math
            _FP8_BLOCK = 128
            self.w13_weight_scale_inv = nn.Parameter(torch.ones(
                self.n_local_experts,
                math.ceil(2 * moe_intermediate / _FP8_BLOCK),
                math.ceil(self.hidden_size / _FP8_BLOCK),
                dtype=torch.float32,
            ), requires_grad=False)
            self.w13_weight_scale_inv.weight_loader = self._w13_scale_loader
            self.w2_weight_scale_inv = nn.Parameter(torch.ones(
                self.n_local_experts,
                math.ceil(self.hidden_size / _FP8_BLOCK),
                math.ceil(moe_intermediate / _FP8_BLOCK),
                dtype=torch.float32,
            ), requires_grad=False)
            self.w2_weight_scale_inv.weight_loader = self._w2_scale_loader
        else:
            self.w13_weight_scale_inv = None
            self.w2_weight_scale_inv = None

        self.grouped_topk = GroupedSigmoidTopK()
        self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        local_id = expert_id - self.expert_start
        if local_id < 0 or local_id >= self.n_local_experts:
            return
        N = param.shape[1] // 2
        offset = 0 if is_w1 else N
        param.data[local_id, offset:offset + N, :].copy_(loaded_weight)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        local_id = expert_id - self.expert_start
        if local_id < 0 or local_id >= self.n_local_experts:
            return
        param.data[local_id].copy_(loaded_weight)

    def _w13_scale_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        local_id = expert_id - self.expert_start
        if local_id < 0 or local_id >= self.n_local_experts:
            return
        N = param.shape[1] // 2
        offset = 0 if is_w1 else N
        param.data[local_id, offset:offset + N, :].copy_(loaded_weight)

    def _w2_scale_loader(self, param, loaded_weight, expert_id: int):
        local_id = expert_id - self.expert_start
        if local_id < 0 or local_id >= self.n_local_experts:
            return
        param.data[local_id].copy_(loaded_weight)

    def _run_routed_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        topk_weights, topk_ids = self.grouped_topk(
            router_logits, self.top_k,
            n_group=self.n_group,
            topk_group=self.topk_group,
            e_score_correction_bias=self.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            renormalize=self.norm_topk_prob,
        )

        if self.ep_size > 1:
            local_mask = (topk_ids >= self.expert_start) & (topk_ids < self.expert_end)
            topk_ids = torch.where(
                local_mask, topk_ids - self.expert_start, torch.zeros_like(topk_ids),
            )
            topk_weights = torch.where(
                local_mask, topk_weights, torch.zeros_like(topk_weights),
            )

        return self.fused_experts(
            hidden_states, self.w13, self.w2,
            topk_weights, topk_ids, self.n_local_experts,
            w13_scale=self.w13_weight_scale_inv,
            w2_scale=self.w2_weight_scale_inv,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        shared_out = None
        if self.shared_experts is not None:
            shared_out = self.shared_experts(hidden_states)

        router_logits = self.gate(hidden_states)

        ep_group = get_ep_group()
        use_ep = self.ep_size > 1 and ep_group is not None

        if not use_ep:
            routed_out = self._run_routed_experts(hidden_states, router_logits)
        else:
            routed_out = self._chunked_ep_forward(
                hidden_states, router_logits, ep_group,
            )

        if shared_out is not None:
            out = routed_out + shared_out
        else:
            out = routed_out

        return out.view(orig_shape)

    def _chunked_ep_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        ep_group,
    ) -> torch.Tensor:
        """Process MoE with EP, chunking only for large batches.

        For small batches (decode), directly all_gather with actual size.
        For large batches (prefill), chunk into _MOE_DP_CHUNK_SIZE pieces.
        All ranks synchronize on max_n via all_reduce(MAX) to ensure
        matching collective counts.
        """
        import torch.distributed as dist

        local_n = hidden_states.size(0)
        ep_size = self.ep_size
        D = self.hidden_size
        R = router_logits.size(1)

        global _ep_cached_max_n
        if _ep_cached_max_n is not None:
            max_n = _ep_cached_max_n
        else:
            max_n_t = torch.tensor([local_n], dtype=torch.int64,
                                   device=hidden_states.device)
            dist.all_reduce(max_n_t, op=dist.ReduceOp.MAX, group=ep_group)
            max_n = int(max_n_t.item())

        if max_n <= _MOE_DP_CHUNK_SIZE:
            return self._ep_forward_small(
                hidden_states, router_logits, ep_group, local_n, max_n)
        else:
            return self._ep_forward_chunked(
                hidden_states, router_logits, ep_group, local_n, max_n)

    def _ep_forward_small(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        ep_group,
        local_n: int,
        max_n: int,
    ) -> torch.Tensor:
        """EP dispatch/combine for small batches (no chunking)."""
        import torch.distributed as dist
        ep_size = self.ep_size
        D = self.hidden_size
        R = router_logits.size(1)

        if local_n < max_n:
            h_pad = hidden_states.new_zeros(max_n - local_n, D)
            h_in = torch.cat([hidden_states, h_pad], dim=0)
            r_pad = router_logits.new_zeros(max_n - local_n, R)
            r_in = torch.cat([router_logits, r_pad], dim=0)
        else:
            h_in = hidden_states.contiguous()
            r_in = router_logits.contiguous()

        gathered_h = [torch.empty_like(h_in) for _ in range(ep_size)]
        dist.all_gather(gathered_h, h_in, group=ep_group)
        gathered_r = [torch.empty_like(r_in) for _ in range(ep_size)]
        dist.all_gather(gathered_r, r_in, group=ep_group)

        all_h = torch.cat(gathered_h, dim=0)
        all_r = torch.cat(gathered_r, dim=0)

        out = self._run_routed_experts(all_h, all_r)

        rs_out = torch.empty(max_n, D, dtype=out.dtype, device=out.device)
        dist.reduce_scatter_tensor(rs_out, out, group=ep_group)
        return rs_out[:local_n]

    def _ep_forward_chunked(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        ep_group,
        local_n: int,
        max_n: int,
    ) -> torch.Tensor:
        """EP dispatch/combine for large batches with chunking."""
        import torch.distributed as dist

        chunk_size = _MOE_DP_CHUNK_SIZE
        ep_size = self.ep_size
        D = self.hidden_size
        R = router_logits.size(1)
        total_gathered = ep_size * chunk_size
        num_chunks = (max_n + chunk_size - 1) // chunk_size

        need_init = (
            not hasattr(self, '_ep_bufs_initialized')
            or self._ep_buf_h.size(0) < total_gathered
        )
        if need_init:
            dev = hidden_states.device
            dt_h = hidden_states.dtype
            dt_r = router_logits.dtype
            self._ep_buf_h = torch.empty(total_gathered, D, dtype=dt_h, device=dev)
            self._ep_buf_r = torch.empty(total_gathered, R, dtype=dt_r, device=dev)
            self._ep_chunk_h = torch.empty(chunk_size, D, dtype=dt_h, device=dev)
            self._ep_chunk_r = torch.empty(chunk_size, R, dtype=dt_r, device=dev)
            self._ep_rs_out = torch.empty(chunk_size, D, dtype=dt_h, device=dev)
            self._ep_gather_h = [
                self._ep_buf_h[i * chunk_size:(i + 1) * chunk_size]
                for i in range(ep_size)
            ]
            self._ep_gather_r = [
                self._ep_buf_r[i * chunk_size:(i + 1) * chunk_size]
                for i in range(ep_size)
            ]
            self._ep_bufs_initialized = True

        result = hidden_states.new_zeros(local_n, D)

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min(c_start + chunk_size, local_n)
            actual = max(c_end - c_start, 0)

            h_chunk = self._ep_chunk_h
            r_chunk = self._ep_chunk_r
            h_chunk.zero_()
            r_chunk.zero_()
            if actual > 0:
                h_chunk[:actual].copy_(hidden_states[c_start:c_end])
                r_chunk[:actual].copy_(router_logits[c_start:c_end])

            dist.all_gather(self._ep_gather_h, h_chunk, group=ep_group)
            dist.all_gather(self._ep_gather_r, r_chunk, group=ep_group)

            chunk_out = self._run_routed_experts(self._ep_buf_h, self._ep_buf_r)

            dist.reduce_scatter_tensor(self._ep_rs_out, chunk_out, group=ep_group)

            if actual > 0:
                result[c_start:c_end] = self._ep_rs_out[:actual]

        return result
