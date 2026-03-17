"""Mamba v1 mixer: selective state space model with causal conv1d.

Flow: in_proj → [x, z] → causal_conv1d(x) → x_proj → [dt, B, C] →
      dt_proj → selective_scan(x, dt, A, B, C, D, z) → out_proj

Weight names match HuggingFace checkpoint:
  mixer.in_proj.weight        [2*intermediate_size, hidden_size]
  mixer.conv1d.weight         [intermediate_size, 1, conv_kernel]
  mixer.conv1d.bias           [intermediate_size]
  mixer.x_proj.weight         [dt_rank + 2*state_size, intermediate_size]
  mixer.dt_proj.weight        [intermediate_size, dt_rank]
  mixer.dt_proj.bias          [intermediate_size]
  mixer.A_log                 [intermediate_size, state_size]
  mixer.D                     [intermediate_size]
  mixer.out_proj.weight       [hidden_size, intermediate_size]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

selective_state_update = None


class MambaMixer(nn.Module):
    """Mamba selective scan mixer block."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.time_step_rank = config.time_step_rank

        self.in_proj = nn.Linear(
            config.hidden_size, 2 * self.intermediate_size, bias=config.use_bias,
        )

        # Depthwise conv1d — weight shape [D, 1, kernel] matches checkpoint
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            bias=config.use_conv_bias,
            padding=config.conv_kernel - 1,
        )

        self.x_proj = nn.Linear(
            self.intermediate_size,
            self.time_step_rank + 2 * self.state_size,
            bias=False,
        )

        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # A_log and D: loaded directly from checkpoint
        self.A_log = nn.Parameter(torch.empty(self.intermediate_size, self.state_size))
        self.D = nn.Parameter(torch.empty(self.intermediate_size))

        self.out_proj = nn.Linear(
            self.intermediate_size, config.hidden_size, bias=config.use_bias,
        )

    def forward(self, hidden_states, cache_params=None, cache_position=None):
        B, T, _ = hidden_states.shape

        # in_proj → x, z (gate)
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        if T > 1:
            return self._prefill(x, z, B, T, cache_params)
        else:
            return self._decode(x, z, B, cache_params)

    def _prefill(self, x, z, B, T, cache_params):
        conv_weight = self.conv1d.weight.squeeze(1)  # [D, kernel]
        x = x.transpose(1, 2).contiguous()  # [B, D, T]

        # Save conv state: last conv_kernel pre-conv values
        if cache_params is not None:
            if T >= self.conv_kernel_size:
                cache_params.conv_states[self.layer_idx].copy_(
                    x[:, :, -self.conv_kernel_size:]
                )
            else:
                cache_params.conv_states[self.layer_idx].zero_()
                cache_params.conv_states[self.layer_idx][:, :, -T:].copy_(x)

        x = causal_conv1d_fn(x, conv_weight, self.conv1d.bias, activation="silu")
        x = x.transpose(1, 2)  # [B, T, D]

        # x_proj → dt, B_ssm, C_ssm
        x_dbl = self.x_proj(x)
        dt, B_ssm, C_ssm = x_dbl.split(
            [self.time_step_rank, self.state_size, self.state_size], dim=-1,
        )
        dt = self.dt_proj(dt)  # [B, T, D]

        # Selective scan
        A = -torch.exp(self.A_log.float())
        y, last_state = selective_scan_fn(
            x.transpose(1, 2).contiguous(),
            dt.transpose(1, 2).contiguous(),
            A,
            B_ssm.transpose(1, 2).contiguous(),
            C_ssm.transpose(1, 2).contiguous(),
            D=self.D.float(),
            z=z.transpose(1, 2).contiguous(),
            delta_softplus=True,
            return_last_state=True,
        )
        # y: [B, D, T], last_state: [B, D, N]

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(last_state)

        return self.out_proj(y.transpose(1, 2))

    def _decode(self, x, z, B, cache_params):
        x = x.squeeze(1)  # [B, D]
        z = z.squeeze(1)

        # Conv1d update
        conv_weight = self.conv1d.weight.squeeze(1)
        x = causal_conv1d_update(
            x, cache_params.conv_states[self.layer_idx],
            conv_weight, self.conv1d.bias, activation="silu",
        )

        # x_proj → dt, B_ssm, C_ssm
        x_dbl = self.x_proj(x)
        dt, B_ssm, C_ssm = x_dbl.split(
            [self.time_step_rank, self.state_size, self.state_size], dim=-1,
        )
        dt = self.dt_proj(dt)  # [B, D]

        A = -torch.exp(self.A_log.float())

        if selective_state_update is not None:
            y = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                x, dt, A, B_ssm, C_ssm,
                D=self.D, z=z, dt_softplus=True,
            )
        else:
            # Manual SSM step
            dt = F.softplus(dt)
            ssm_state = cache_params.ssm_states[self.layer_idx]
            dA = torch.exp(dt.float().unsqueeze(-1) * A)
            dB = dt.float().unsqueeze(-1) * B_ssm.float().unsqueeze(1)
            ssm_state.copy_(
                (ssm_state.float() * dA + x.float().unsqueeze(-1) * dB).to(ssm_state.dtype)
            )
            y = torch.einsum("bdn,bn->bd", ssm_state.float(), C_ssm.float())
            y = y + self.D.float() * x.float()
            y = (y * F.silu(z.float())).to(x.dtype)

        return self.out_proj(y.unsqueeze(1))
