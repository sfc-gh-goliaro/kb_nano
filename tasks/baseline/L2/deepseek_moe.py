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
from ..L2.fused_experts import FusedExperts, use_flashinfer_cutlass

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
        self._use_flashinfer = use_flashinfer_cutlass()

        self._shared_stream = (
            torch.cuda.Stream() if self.shared_experts is not None
            and not os.environ.get("KB_NANO_DISABLE_SHARED_EXPERTS_STREAM", "0") == "1"
            else None
        )

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
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self._use_flashinfer:
            return self.fused_experts(
                hidden_states, self.w13, self.w2,
                topk_weights, topk_ids, self.n_local_experts,
                w13_scale=self.w13_weight_scale_inv,
                w2_scale=self.w2_weight_scale_inv,
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
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

    def _compute_routing(self, router_logits: torch.Tensor):
        return self.grouped_topk(
            router_logits, self.top_k,
            n_group=self.n_group,
            topk_group=self.topk_group,
            e_score_correction_bias=self.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            renormalize=self.norm_topk_prob,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        num_tokens = hidden_states.size(0)

        capturing = torch.cuda.is_current_stream_capturing()
        use_overlap = (
            self._shared_stream is not None
            and self.shared_experts is not None
            and num_tokens <= 256
            and not capturing
        )

        if use_overlap:
            shared_input = hidden_states.clone()
            shared_input.record_stream(self._shared_stream)
            main_stream = torch.cuda.current_stream()
            self._shared_stream.wait_stream(main_stream)
            with torch.cuda.stream(self._shared_stream):
                shared_out = self.shared_experts(shared_input)
        elif self.shared_experts is not None:
            shared_out = self.shared_experts(hidden_states)

        router_logits = self.gate(hidden_states)
        topk_weights, topk_ids = self._compute_routing(router_logits)

        ep_group = get_ep_group()
        use_ep = self.ep_size > 1 and ep_group is not None

        if not use_ep:
            routed_out = self._run_routed_experts(
                hidden_states, topk_weights, topk_ids)
        else:
            routed_out = self._chunked_ep_forward(
                hidden_states, topk_weights, topk_ids, ep_group,
            )

        if use_overlap:
            main_stream.wait_stream(self._shared_stream)

        if self.shared_experts is not None:
            out = routed_out + shared_out
        else:
            out = routed_out

        return out.view(orig_shape)

    def _chunked_ep_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        ep_group,
    ) -> torch.Tensor:
        """Process MoE with EP, chunking only for large batches.

        Routing is computed locally before gather. We gather hidden_states
        and post-routing topk_weights/topk_ids (much smaller than full
        router_logits).
        """
        import torch.distributed as dist

        local_n = hidden_states.size(0)
        ep_size = self.ep_size
        D = self.hidden_size
        top_k = topk_ids.size(1)

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
                hidden_states, topk_weights, topk_ids,
                ep_group, local_n, max_n)
        else:
            return self._ep_forward_chunked(
                hidden_states, topk_weights, topk_ids,
                ep_group, local_n, max_n)

    def _ep_forward_small(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        ep_group,
        local_n: int,
        max_n: int,
    ) -> torch.Tensor:
        """EP dispatch/combine for small batches (no chunking).

        Uses pre-allocated buffers when available.
        """
        import torch.distributed as dist
        ep_size = self.ep_size
        D = self.hidden_size
        top_k = topk_ids.size(1)
        total = ep_size * max_n

        have_bufs = getattr(self, '_ep_bufs_initialized', False)

        if have_bufs:
            h_in = self._ep_chunk_h[:max_n]
            w_in = self._ep_chunk_w[:max_n]
            i_in = self._ep_chunk_i[:max_n]
            h_in[:local_n].copy_(hidden_states[:local_n])
            w_in[:local_n].copy_(topk_weights[:local_n])
            i_in[:local_n].copy_(topk_ids[:local_n])
            if local_n < max_n:
                h_in[local_n:max_n].zero_()
                w_in[local_n:max_n].zero_()
                i_in[local_n:max_n].zero_()

            gather_h = [t[:max_n] for t in self._ep_gather_h]
            gather_w = [t[:max_n] for t in self._ep_gather_w]
            gather_i = [t[:max_n] for t in self._ep_gather_i]
        else:
            if local_n < max_n:
                pad = max_n - local_n
                h_in = torch.cat([hidden_states,
                                  hidden_states.new_zeros(pad, D)], dim=0)
                w_in = torch.cat([topk_weights,
                                  topk_weights.new_zeros(pad, top_k)], dim=0)
                i_in = torch.cat([topk_ids,
                                  topk_ids.new_zeros(pad, top_k)], dim=0)
            else:
                h_in = hidden_states.contiguous()
                w_in = topk_weights.contiguous()
                i_in = topk_ids.contiguous()

            gather_h = [torch.empty_like(h_in) for _ in range(ep_size)]
            gather_w = [torch.empty_like(w_in) for _ in range(ep_size)]
            gather_i = [torch.empty_like(i_in) for _ in range(ep_size)]

        dist.all_gather(gather_h, h_in, group=ep_group)
        dist.all_gather(gather_w, w_in, group=ep_group)
        dist.all_gather(gather_i, i_in, group=ep_group)

        if have_bufs:
            all_h = self._ep_buf_h[:total]
            all_w = self._ep_buf_w[:total]
            all_i = self._ep_buf_i[:total]
        else:
            all_h = torch.cat(gather_h, dim=0)
            all_w = torch.cat(gather_w, dim=0)
            all_i = torch.cat(gather_i, dim=0)

        out = self._run_routed_experts(all_h, all_w, all_i)

        if have_bufs:
            rs_out = self._ep_rs_out[:max_n]
        else:
            rs_out = torch.empty(max_n, D, dtype=out.dtype, device=out.device)
        dist.reduce_scatter_tensor(rs_out, out, group=ep_group)
        return rs_out[:local_n]

    def _ep_forward_chunked(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        ep_group,
        local_n: int,
        max_n: int,
    ) -> torch.Tensor:
        """EP dispatch/combine for large batches with chunking."""
        import torch.distributed as dist

        chunk_size = _MOE_DP_CHUNK_SIZE
        ep_size = self.ep_size
        D = self.hidden_size
        top_k = topk_ids.size(1)
        total_gathered = ep_size * chunk_size
        num_chunks = (max_n + chunk_size - 1) // chunk_size

        need_init = (
            not hasattr(self, '_ep_bufs_initialized')
            or self._ep_buf_h.size(0) < total_gathered
        )
        if need_init:
            dev = hidden_states.device
            dt_h = hidden_states.dtype
            dt_w = topk_weights.dtype
            dt_i = topk_ids.dtype
            self._ep_buf_h = torch.empty(total_gathered, D, dtype=dt_h, device=dev)
            self._ep_buf_w = torch.empty(total_gathered, top_k, dtype=dt_w, device=dev)
            self._ep_buf_i = torch.empty(total_gathered, top_k, dtype=dt_i, device=dev)
            self._ep_chunk_h = torch.empty(chunk_size, D, dtype=dt_h, device=dev)
            self._ep_chunk_w = torch.empty(chunk_size, top_k, dtype=dt_w, device=dev)
            self._ep_chunk_i = torch.empty(chunk_size, top_k, dtype=dt_i, device=dev)
            self._ep_rs_out = torch.empty(chunk_size, D, dtype=dt_h, device=dev)
            self._ep_gather_h = [
                self._ep_buf_h[i * chunk_size:(i + 1) * chunk_size]
                for i in range(ep_size)
            ]
            self._ep_gather_w = [
                self._ep_buf_w[i * chunk_size:(i + 1) * chunk_size]
                for i in range(ep_size)
            ]
            self._ep_gather_i = [
                self._ep_buf_i[i * chunk_size:(i + 1) * chunk_size]
                for i in range(ep_size)
            ]
            self._ep_bufs_initialized = True

        result = hidden_states.new_zeros(local_n, D)

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min(c_start + chunk_size, local_n)
            actual = max(c_end - c_start, 0)

            h_chunk = self._ep_chunk_h
            w_chunk = self._ep_chunk_w
            i_chunk = self._ep_chunk_i
            h_chunk.zero_()
            w_chunk.zero_()
            i_chunk.zero_()
            if actual > 0:
                h_chunk[:actual].copy_(hidden_states[c_start:c_end])
                w_chunk[:actual].copy_(topk_weights[c_start:c_end])
                i_chunk[:actual].copy_(topk_ids[c_start:c_end])

            dist.all_gather(self._ep_gather_h, h_chunk, group=ep_group)
            dist.all_gather(self._ep_gather_w, w_chunk, group=ep_group)
            dist.all_gather(self._ep_gather_i, i_chunk, group=ep_group)

            chunk_out = self._run_routed_experts(
                self._ep_buf_h, self._ep_buf_w, self._ep_buf_i)

            dist.reduce_scatter_tensor(self._ep_rs_out, chunk_out, group=ep_group)

            if actual > 0:
                result[c_start:c_end] = self._ep_rs_out[:actual]

        return result
