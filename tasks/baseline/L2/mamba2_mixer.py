"""Mamba2 mixer: structured state space duality (SSD) with multi-head SSM.

Flow: in_proj → [gate, x+B+C, dt] → causal_conv1d(x+B+C) → split(x, B, C) →
      mamba_chunk_scan(x, dt, A, B, C, D) → rms_norm(y) * silu(gate) → out_proj

Weight names match HuggingFace checkpoint:
  mixer.in_proj.weight        [intermediate_size + conv_dim + num_heads, hidden_size]
  mixer.conv1d.weight         [conv_dim, 1, conv_kernel]
  mixer.conv1d.bias           [conv_dim]
  mixer.A_log                 [num_heads]
  mixer.D                     [num_heads]
  mixer.dt_bias               [num_heads]
  mixer.norm.weight           [intermediate_size]
  mixer.out_proj.weight       [hidden_size, intermediate_size]

where conv_dim = intermediate_size + 2 * n_groups * state_size
      intermediate_size = num_heads * head_dim
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.selective_state_update import selective_state_update


class Mamba2Mixer(nn.Module):
    """Mamba2 SSD mixer block."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = getattr(config, "num_heads", 128)
        self.head_dim = config.head_dim
        self.intermediate_size = self.num_heads * self.head_dim
        self.state_size = config.state_size
        self.n_groups = config.n_groups
        self.conv_kernel_size = config.conv_kernel
        self.chunk_size = getattr(config, "chunk_size", 256)

        # conv_dim = intermediate_size + 2 * n_groups * state_size
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.state_size

        # in_proj: projects hidden_size → gate + (x+B+C) + dt
        # Total output: intermediate_size + conv_dim + num_heads
        in_proj_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(config.hidden_size, in_proj_size, bias=False)

        # Depthwise conv1d on x+B+C concatenated
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            bias=getattr(config, "use_conv_bias", True),
            padding=config.conv_kernel - 1,
        )

        # Per-head parameters
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.D = nn.Parameter(torch.empty(self.num_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads))

        # Gated RMS norm on SSM output (per-group normalization)
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(self.intermediate_size))

        self.out_proj = nn.Linear(
            self.intermediate_size, config.hidden_size, bias=False,
        )

    def forward(self, hidden_states, cache_params=None, cache_position=None):
        B, T, _ = hidden_states.shape

        # in_proj → gate, hidden_states_B_C, dt
        projected = self.in_proj(hidden_states)
        gate, hidden_states_B_C, dt = projected.split(
            [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1,
        )

        if T > 1:
            return self._prefill(gate, hidden_states_B_C, dt, B, T, cache_params)
        else:
            return self._decode(gate, hidden_states_B_C, dt, B, cache_params)

    def _prefill(self, gate, hidden_states_B_C, dt, B, T, cache_params):
        conv_weight = self.conv1d.weight.squeeze(1)  # [conv_dim, kernel]

        # Transpose for causal_conv1d: [B, conv_dim, T]
        hidden_states_B_C = hidden_states_B_C.transpose(1, 2).contiguous()

        # Save conv state: last conv_kernel pre-conv values
        if cache_params is not None:
            if T >= self.conv_kernel_size:
                cache_params.conv_states[self.layer_idx].copy_(
                    hidden_states_B_C[:, :, -self.conv_kernel_size:]
                )
            else:
                cache_params.conv_states[self.layer_idx].zero_()
                cache_params.conv_states[self.layer_idx][:, :, -T:].copy_(
                    hidden_states_B_C
                )

        # Causal conv1d with SiLU activation
        hidden_states_B_C = causal_conv1d_fn(
            hidden_states_B_C, conv_weight, self.conv1d.bias, activation="silu",
        )
        hidden_states_B_C = hidden_states_B_C.transpose(1, 2)  # [B, T, conv_dim]

        # Split into x, B_ssm, C_ssm
        x, B_ssm, C_ssm = hidden_states_B_C.split(
            [self.intermediate_size, self.n_groups * self.state_size,
             self.n_groups * self.state_size],
            dim=-1,
        )

        # Reshape for multi-head SSM
        # x: [B, T, num_heads, head_dim]
        x = x.view(B, T, self.num_heads, self.head_dim)
        # B_ssm: [B, T, n_groups, state_size]
        B_ssm = B_ssm.view(B, T, self.n_groups, self.state_size)
        # C_ssm: [B, T, n_groups, state_size]
        C_ssm = C_ssm.view(B, T, self.n_groups, self.state_size)

        A = -torch.exp(self.A_log.float())

        # mamba_chunk_scan_combined
        y, last_state = mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B_ssm,
            C_ssm,
            chunk_size=self.chunk_size,
            D=self.D.float(),
            z=None,
            dt_bias=self.dt_bias.float(),
            dt_softplus=True,
            return_final_states=True,
        )
        # y: [B, T, num_heads, head_dim]
        # last_state: [B, num_heads, head_dim, state_size]

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(last_state)

        # Flatten heads: [B, T, intermediate_size]
        y = y.view(B, T, self.intermediate_size)

        # Gated RMS norm: rms_norm(y) * silu(gate)
        y = self._gated_rms_norm(y, gate)

        return self.out_proj(y)

    def _decode(self, gate, hidden_states_B_C, dt, B, cache_params):
        gate = gate.squeeze(1)  # [B, intermediate_size]
        hidden_states_B_C = hidden_states_B_C.squeeze(1)  # [B, conv_dim]
        dt = dt.squeeze(1)  # [B, num_heads]

        # Conv1d update
        conv_weight = self.conv1d.weight.squeeze(1)
        hidden_states_B_C = causal_conv1d_update(
            hidden_states_B_C,
            cache_params.conv_states[self.layer_idx],
            conv_weight,
            self.conv1d.bias,
            activation="silu",
        )

        # Split into x, B_ssm, C_ssm
        x, B_ssm, C_ssm = hidden_states_B_C.split(
            [self.intermediate_size, self.n_groups * self.state_size,
             self.n_groups * self.state_size],
            dim=-1,
        )

        # Reshape for multi-head SSM
        x = x.view(B, self.num_heads, self.head_dim)
        B_ssm = B_ssm.view(B, self.n_groups, self.state_size)
        C_ssm = C_ssm.view(B, self.n_groups, self.state_size)

        A = -torch.exp(self.A_log.float())
        # A needs shape [num_heads, head_dim, state_size] for selective_state_update
        A = A[:, None, None].expand(-1, self.head_dim, self.state_size).to(torch.float32)
        dt_expanded = dt[:, :, None].expand(-1, -1, self.head_dim)
        dt_bias = self.dt_bias[:, None].expand(-1, self.head_dim)
        D = self.D[:, None].expand(-1, self.head_dim)

        ssm_state = cache_params.ssm_states[self.layer_idx]

        y = selective_state_update(
            ssm_state,
            x,
            dt_expanded,
            A,
            B_ssm,
            C_ssm,
            D=D,
            z=None,
            dt_bias=dt_bias,
            dt_softplus=True,
        )
        # y: [B, num_heads, head_dim]
        y = y.view(B, self.intermediate_size)

        # Gated RMS norm
        y = self._gated_rms_norm(y, gate)

        return self.out_proj(y).unsqueeze(1)

    def _gated_rms_norm(self, x, gate):
        """Apply gated RMS norm: rms_norm(x, per_group) * silu(gate).

        x and gate both have shape [..., intermediate_size].
        Normalization is per-group where group_size = intermediate_size // n_groups.
        """
        input_dtype = x.dtype
        # Apply gate first: x = x * silu(gate)
        x = x * F.silu(gate.float())

        # Per-group RMS normalization
        group_size = self.intermediate_size // self.n_groups
        shape = x.shape
        x = x.view(*shape[:-1], self.n_groups, group_size)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-5)
        x = x.view(*shape)

        return (self.norm.weight * x).to(input_dtype)
