"""KDA (Kimi Delta Attention) layer.

Implements the full KDA block used in Kimi-Linear:
  x -> q/k/v proj -> causal conv1d (SiLU) -> Delta-Net recurrence
    -> RMSNorm + sigmoid gate -> o proj

Gate computation:
  g1 = f_b(f_a(x))  -> fused_kda_gate(g1, A_log, dt_bias)  (forget gate)
  beta = sigmoid(b_proj(x))                                  (learning rate)
  g2 = g_b(g_a(x))                                           (output gate)

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
import torch.nn.functional as F

from fla.modules.convolution import causal_conv1d
from fla.ops.kda import fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate

from ....infra.tp import _tp_size, _tp_rank
from .parallel_linear import ColumnParallelLinear, RowParallelLinear


class _SimpleRMSNorm(nn.Module):
    """Lightweight RMSNorm with .weight attribute for weight loading."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.weight.weight_loader = lambda p, w: p.data.copy_(w)
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps) * self.weight).to(x.dtype)


class ReplicatedLinear(nn.Module):
    """Linear layer replicated across TP ranks (no sharding)."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = lambda p, w: p.data.copy_(w)
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def _causal_conv1d_prefill(x, weight):
    """Causal 1D depthwise convolution with SiLU for full sequence.

    Args:
        x: [B, T, D]  projected input (pre-conv)
        weight: [D, K]  depthwise conv kernel
    Returns:
        output: [B, T, D]
    """
    D = x.shape[2]
    K = weight.shape[1]
    xt = x.transpose(1, 2).contiguous()  # [B, D, T]
    xt = F.pad(xt, (K - 1, 0))
    out = F.conv1d(xt, weight.unsqueeze(1).to(xt.dtype), groups=D)
    return F.silu(out).transpose(1, 2)


def _causal_conv1d_decode(x, conv_state, weight):
    """Single-step causal conv1d for decode.

    Args:
        x: [B, 1, D]  single timestep projected input
        conv_state: [B, D, K-1]  previous state (last K-1 projected values)
        weight: [D, K]
    Returns:
        output: [B, 1, D]
        new_state: [B, D, K-1]
    """
    x_flat = x.squeeze(1)  # [B, D]
    # Append new value: [B, D, K-1] + [B, D, 1] -> take last K-1
    new_state = torch.cat([conv_state[:, :, 1:], x_flat.unsqueeze(-1)], dim=-1)
    # Full window: [state..., current] = [B, D, K]
    window = torch.cat([new_state, x_flat.unsqueeze(-1)], dim=-1)
    # Wait, we should include the new value in the conv window.
    # State has K-1 past values, plus current = K total values.
    full_window = torch.cat([conv_state, x_flat.unsqueeze(-1)], dim=-1)  # [B, D, K]
    out = (full_window * weight.unsqueeze(0).to(full_window.dtype)).sum(-1)  # [B, D]
    out = F.silu(out)
    # New state: drop oldest, append current
    new_state = torch.cat([conv_state[:, :, 1:], x_flat.unsqueeze(-1)], dim=-1)
    return out.unsqueeze(1), new_state


def _compute_kda_gate(g, A_log, dt_bias, num_local_heads, head_dim):
    """Compute KDA forget gate in log space.

    g: [B, T, H_local*D] raw gate projection output
    A_log: [1, 1, H_local, 1]
    dt_bias: [H_local*D]
    Returns: [B, T, H_local, D] log-space gate (negative values)
    """
    g = g + dt_bias
    g = g.unflatten(-1, (num_local_heads, head_dim))  # [B, T, H_local, D]
    dt = F.softplus(g.float())
    A = -A_log.exp()  # [1, 1, H_local, 1], negative
    g_log = dt * A  # [B, T, H_local, D], negative
    return g_log  # [B, T, H_local, D]


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

        # Causal conv1d weights: use ColumnParallelLinear as container
        # Checkpoint has [channels, 1, kernel_size] (nn.Conv1d), we store [D_local, K].
        self.q_conv1d = ColumnParallelLinear(conv_kernel_size, projection_size)
        self.q_conv1d.weight.weight_loader = self._conv_weight_loader
        self.k_conv1d = ColumnParallelLinear(conv_kernel_size, projection_size)
        self.k_conv1d.weight.weight_loader = self._conv_weight_loader
        self.v_conv1d = ColumnParallelLinear(conv_kernel_size, projection_size)
        self.v_conv1d.weight.weight_loader = self._conv_weight_loader

        # Forget gate: f_a (replicated) -> f_b (column parallel)
        self.f_a_proj = ReplicatedLinear(hidden_size, head_dim)
        self.f_b_proj = ColumnParallelLinear(head_dim, projection_size)

        # Decay parameters
        self.dt_bias = nn.Parameter(torch.empty(local_projection, dtype=torch.float32))
        self.dt_bias.weight_loader = self._sharded_dim0
        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        self.A_log.weight_loader = self._sharded_dim2

        # Beta (learning rate) gate
        self.b_proj = ColumnParallelLinear(hidden_size, num_heads)

        # Output gate: g_a (replicated) -> g_b (column parallel)
        self.g_a_proj = ReplicatedLinear(hidden_size, head_dim)
        self.g_b_proj = ColumnParallelLinear(head_dim, projection_size)

        # Output norm: RMSNorm with .weight (matches checkpoint o_norm.weight)
        self.o_norm = _SimpleRMSNorm(head_dim)

    @staticmethod
    def _conv_weight_loader(param, loaded_weight):
        # Checkpoint stores [channels, 1, kernel_size]; squeeze to [channels, kernel_size]
        if loaded_weight.dim() == 3:
            loaded_weight = loaded_weight.squeeze(1)
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    @staticmethod
    def _sharded_dim0(param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    @staticmethod
    def _sharded_dim2(param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(2)
        param.data.copy_(loaded_weight.narrow(2, rank * shard, shard))

    def _output_norm_gate(self, attn_out, g2):
        """RMSNorm(attn_out) * sigmoid(g2).

        attn_out, g2: [B, T, H_local, D]
        """
        normed = self.o_norm(attn_out)
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
        is_decode = (layer_state is not None and T == 1)

        # Projections (before conv)
        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)

        # Gates (independent of conv)
        beta = self.b_proj(hidden_states).float().sigmoid()  # [B, T, H_local]
        g1_raw = self.f_b_proj(self.f_a_proj(hidden_states))  # [B, T, D_local]
        g2_raw = self.g_b_proj(self.g_a_proj(hidden_states))  # [B, T, D_local]

        # Causal conv1d using FLA's kernel
        output_final_state = (layer_state is not None)
        conv_cache_q = layer_state.get("conv_q") if layer_state is not None else None
        conv_cache_k = layer_state.get("conv_k") if layer_state is not None else None
        conv_cache_v = layer_state.get("conv_v") if layer_state is not None else None

        q, conv_state_q = causal_conv1d(
            q_proj, self.q_conv1d.weight, activation='silu',
            initial_state=conv_cache_q, output_final_state=output_final_state,
        )
        k, conv_state_k = causal_conv1d(
            k_proj, self.k_conv1d.weight, activation='silu',
            initial_state=conv_cache_k, output_final_state=output_final_state,
        )
        v, conv_state_v = causal_conv1d(
            v_proj, self.v_conv1d.weight, activation='silu',
            initial_state=conv_cache_v, output_final_state=output_final_state,
        )
        if layer_state is not None:
            layer_state["conv_q"] = conv_state_q
            layer_state["conv_k"] = conv_state_k
            layer_state["conv_v"] = conv_state_v

        # Reshape for recurrence: [B, T, H, D] (FLA convention)
        q = q.view(B, T, self.local_num_heads, self.head_dim)
        k = k.view(B, T, self.local_num_heads, self.head_dim)
        v = v.view(B, T, self.local_num_heads, self.head_dim)

        # Compute forget gate in log space: [B, T, H, D] using FLA's kernel
        g1 = g1_raw.view(B, T, self.local_num_heads, self.head_dim)
        g_log = fused_kda_gate(g1, self.A_log, self.dt_bias)

        # Delta-Net recurrence using FLA's fused kernel
        initial_state = layer_state.get("recurrent") if layer_state is not None else None
        attn_out, final_state = fused_recurrent_kda(
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
