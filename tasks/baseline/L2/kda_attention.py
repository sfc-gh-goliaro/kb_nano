"""KDA (Kimi Delta Attention) layer (L2).

Implements the KDA block used in Kimi-Linear:
  x -> q/k/v proj -> causal conv1d (SiLU) -> Delta-Net recurrence
    -> RMSNorm + sigmoid gate -> o proj

Gate computation:
  g1 = f_b(f_a(x))  -> KDAGate(g1, A_log, dt_bias)  (forget gate)
  beta = sigmoid(b_proj(x))                          (learning rate)
  g2 = g_b(g_a(x))                                   (output gate)

Composes only L1 ops (``CausalConv1d``, ``KDARecurrence``, ``KDAGate``,
``RMSNorm``) and the canonical TP linears in ``parallel_linear``; no
external libraries (``fla``, ``flashinfer``, ``vllm``) are imported here.

Weight names match HuggingFace checkpoint convention:
  self_attn.{q,k,v}_proj.weight
  self_attn.{q,k,v}_conv1d.weight
  self_attn.f_a_proj.weight, self_attn.f_b_proj.weight
  self_attn.dt_bias, self_attn.A_log
  self_attn.b_proj.weight
  self_attn.g_a_proj.weight, self_attn.g_b_proj.weight
  self_attn.o_norm.weight
  self_attn.o_proj.weight
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size, _tp_rank
from ..L1.causal_conv1d import CausalConv1d
from ..L1.kda_recurrence import KDAGate, KDARecurrence
from ..L1.rms_norm import RMSNorm
from .parallel_linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


class KDAAttention(nn.Module):
    """Kimi Delta Attention (linear attention with Delta-Net recurrence)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        tp = _tp_size()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.local_num_heads = num_heads // tp
        projection_size = num_heads * head_dim
        local_projection = self.local_num_heads * head_dim
        self.conv_kernel_size = conv_kernel_size
        self.rms_norm_eps = rms_norm_eps

        # Main projections
        self.q_proj = ColumnParallelLinear(hidden_size, projection_size)
        self.k_proj = ColumnParallelLinear(hidden_size, projection_size)
        self.v_proj = ColumnParallelLinear(hidden_size, projection_size)
        self.o_proj = RowParallelLinear(projection_size, hidden_size)

        # Causal conv1d weights: use ColumnParallelLinear as the parameter
        # container (gives us TP sharding on dim 0 for free). The checkpoint
        # stores [channels, 1, kernel] (nn.Conv1d layout); we squeeze to
        # [channels_local, kernel] via the custom loader below.
        self.q_conv1d = ColumnParallelLinear(conv_kernel_size, projection_size)
        self.q_conv1d.weight.weight_loader = self._conv_weight_loader
        self.k_conv1d = ColumnParallelLinear(conv_kernel_size, projection_size)
        self.k_conv1d.weight.weight_loader = self._conv_weight_loader
        self.v_conv1d = ColumnParallelLinear(conv_kernel_size, projection_size)
        self.v_conv1d.weight.weight_loader = self._conv_weight_loader

        # Forget gate: f_a (replicated) -> f_b (column parallel)
        self.f_a_proj = ReplicatedLinear(hidden_size, head_dim, bias=False)
        self.f_b_proj = ColumnParallelLinear(head_dim, projection_size)

        # Decay parameters
        self.dt_bias = nn.Parameter(
            torch.empty(local_projection, dtype=torch.float32)
        )
        self.dt_bias.weight_loader = self._sharded_dim0
        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        self.A_log.weight_loader = self._sharded_dim2

        # Beta (learning rate) gate
        self.b_proj = ColumnParallelLinear(hidden_size, num_heads)

        # Output gate: g_a (replicated) -> g_b (column parallel)
        self.g_a_proj = ReplicatedLinear(hidden_size, head_dim, bias=False)
        self.g_b_proj = ColumnParallelLinear(head_dim, projection_size)

        # Per-head output norm (matches checkpoint o_norm.weight shape [head_dim])
        self.o_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        # L1 ops
        self.q_conv = CausalConv1d()
        self.k_conv = CausalConv1d()
        self.v_conv = CausalConv1d()
        self.kda_gate = KDAGate()
        self.kda_recurrence = KDARecurrence()

    @staticmethod
    def _conv_weight_loader(param, loaded_weight):
        # Checkpoint stores [channels, 1, kernel_size]; squeeze to [channels, K]
        if loaded_weight.dim() == 3:
            loaded_weight = loaded_weight.squeeze(1)
        rank = _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    @staticmethod
    def _sharded_dim0(param, loaded_weight):
        rank = _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    @staticmethod
    def _sharded_dim2(param, loaded_weight):
        rank = _tp_rank()
        shard = param.data.size(2)
        param.data.copy_(loaded_weight.narrow(2, rank * shard, shard))

    def _output_norm_gate(self, attn_out, g2):
        """RMSNorm(attn_out) * sigmoid(g2).

        attn_out, g2: [B, T, H_local, D]. RMSNorm operates per-head_dim
        slice, so we flatten to [-1, D] before the norm and reshape back.
        """
        flat = attn_out.reshape(-1, self.head_dim)
        normed = self.o_norm(flat).view_as(attn_out)
        return normed * torch.sigmoid(g2)

    def forward(self, hidden_states, layer_state=None):
        """
        Args:
            hidden_states: [B, T, hidden_size]
            layer_state: dict with 'conv_q','conv_k','conv_v','recurrent' or None
        Returns:
            output: [B, T, hidden_size]
        """
        B, T, _ = hidden_states.shape

        # Projections (before conv)
        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)

        # Gates (independent of conv)
        beta = self.b_proj(hidden_states).float().sigmoid()  # [B, T, H_local]
        g1_raw = self.f_b_proj(self.f_a_proj(hidden_states))  # [B, T, D_local]
        g2_raw = self.g_b_proj(self.g_a_proj(hidden_states))  # [B, T, D_local]

        # Causal conv1d (L1 op) — handles prefill and decode via initial_state
        output_final_state = (layer_state is not None)
        conv_cache_q = layer_state.get("conv_q") if layer_state is not None else None
        conv_cache_k = layer_state.get("conv_k") if layer_state is not None else None
        conv_cache_v = layer_state.get("conv_v") if layer_state is not None else None

        q, conv_state_q = self.q_conv(
            q_proj, self.q_conv1d.weight,
            initial_state=conv_cache_q,
            output_final_state=output_final_state,
        )
        k, conv_state_k = self.k_conv(
            k_proj, self.k_conv1d.weight,
            initial_state=conv_cache_k,
            output_final_state=output_final_state,
        )
        v, conv_state_v = self.v_conv(
            v_proj, self.v_conv1d.weight,
            initial_state=conv_cache_v,
            output_final_state=output_final_state,
        )
        if layer_state is not None:
            layer_state["conv_q"] = conv_state_q
            layer_state["conv_k"] = conv_state_k
            layer_state["conv_v"] = conv_state_v

        # Reshape for recurrence: [B, T, H, D]
        q = q.view(B, T, self.local_num_heads, self.head_dim)
        k = k.view(B, T, self.local_num_heads, self.head_dim)
        v = v.view(B, T, self.local_num_heads, self.head_dim)

        # Forget gate in log space (L1 op)
        g1 = g1_raw.view(B, T, self.local_num_heads, self.head_dim)
        g_log = self.kda_gate(g1, self.A_log, self.dt_bias)

        # Delta-Net recurrence (L1 op)
        initial_state = layer_state.get("recurrent") if layer_state is not None else None
        attn_out, final_state = self.kda_recurrence(
            q, k, v, g_log, beta,
            initial_state=initial_state,
            output_final_state=(layer_state is not None),
            use_qk_l2norm_in_kernel=True,
        )
        if layer_state is not None and final_state is not None:
            layer_state["recurrent"] = final_state

        # Output gating: RMSNorm + sigmoid
        g2 = g2_raw.view(B, T, self.local_num_heads, self.head_dim)
        attn_out = self._output_norm_gate(attn_out, g2)

        # Output projection
        D_local = self.local_num_heads * self.head_dim
        return self.o_proj(attn_out.reshape(B, T, D_local))
