from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.oasis_rotary import OasisRotaryEmbedding
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L2.oasis_mlp import OasisMLP
from fastkernels.tasks.baseline.L2.oasis_spatial_axial_attention import OasisSpatialAxialAttention
from fastkernels.tasks.baseline.L2.oasis_temporal_axial_attention import OasisTemporalAxialAttention


@triton.jit
def _fused_ln_mod_kernel(
    X_ptr, Out_ptr, Mod_ptr,
    spatial_size, batch_c,
    D: tl.constexpr,
    SHIFT_OFF: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    batch_idx = (row // spatial_size) % batch_c
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x = tl.load(X_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    xn = xc * tl.rsqrt(var + EPS)

    mod_base = batch_idx * (6 * D)
    shift = tl.load(Mod_ptr + mod_base + SHIFT_OFF * D + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(Mod_ptr + mod_base + (SHIFT_OFF + 1) * D + cols, mask=mask, other=0.0).to(tl.float32)

    tl.store(Out_ptr + row * D + cols, xn * (1.0 + scale) + shift, mask=mask)


@triton.jit
def _fused_gate_res_kernel(
    Res_ptr, X_ptr, Mod_ptr, Out_ptr,
    spatial_size, batch_c,
    D: tl.constexpr,
    GATE_OFF: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    batch_idx = (row // spatial_size) % batch_c
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    r = tl.load(Res_ptr + row * D + cols, mask=mask, other=0.0)
    x = tl.load(X_ptr + row * D + cols, mask=mask, other=0.0)
    g = tl.load(Mod_ptr + batch_idx * (6 * D) + GATE_OFF * D + cols, mask=mask, other=0.0)

    tl.store(Out_ptr + row * D + cols, r + g * x, mask=mask)


def _apply_ln_mod(x, mod, spatial_size, batch_c, D, shift_offset, eps=1e-6):
    orig_shape = x.shape
    x_flat = x.contiguous().view(-1, D)
    N = x_flat.shape[0]
    out = torch.empty_like(x_flat)
    BLOCK_D = triton.next_power_of_2(D)
    nw = max(1, min(32, BLOCK_D // 256))
    _fused_ln_mod_kernel[(N,)](
        x_flat, out, mod,
        spatial_size, batch_c,
        D, shift_offset, eps, BLOCK_D,
        num_warps=nw,
    )
    return out.view(orig_shape)


def _apply_gate_res(residual, x, mod, spatial_size, batch_c, D, gate_offset):
    orig_shape = residual.shape
    rf = residual.contiguous().view(-1, D)
    xf = x.contiguous().view(-1, D)
    N = rf.shape[0]
    out = torch.empty_like(rf)
    BLOCK_D = triton.next_power_of_2(D)
    nw = max(1, min(32, BLOCK_D // 256))
    _fused_gate_res_kernel[(N,)](
        rf, xf, mod, out,
        spatial_size, batch_c,
        D, gate_offset, BLOCK_D,
        num_warps=nw,
    )
    return out.view(orig_shape)


class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        is_causal: bool = True,
        spatial_rotary_emb: OasisRotaryEmbedding,
        temporal_rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.s_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_attn = OasisSpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.s_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.t_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_attn = OasisTemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=temporal_rotary_emb,
            is_causal=is_causal,
        )
        self.t_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.t_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        D = self.hidden_size
        batch_c = c.shape[0]
        spatial_size = x.shape[1] * x.shape[2] * x.shape[3]

        c_act = F.silu(c)
        s_mod = F.linear(c_act, self.s_adaLN_modulation[1].weight, self.s_adaLN_modulation[1].bias)
        t_mod = F.linear(c_act, self.t_adaLN_modulation[1].weight, self.t_adaLN_modulation[1].bias)

        normed = _apply_ln_mod(x, s_mod, spatial_size, batch_c, D, 0)
        x = _apply_gate_res(x, self.s_attn(normed), s_mod, spatial_size, batch_c, D, 2)

        normed = _apply_ln_mod(x, s_mod, spatial_size, batch_c, D, 3)
        x = _apply_gate_res(x, self.s_mlp(normed), s_mod, spatial_size, batch_c, D, 5)

        normed = _apply_ln_mod(x, t_mod, spatial_size, batch_c, D, 0)
        x = _apply_gate_res(x, self.t_attn(normed), t_mod, spatial_size, batch_c, D, 2)

        normed = _apply_ln_mod(x, t_mod, spatial_size, batch_c, D, 3)
        x = _apply_gate_res(x, self.t_mlp(normed), t_mod, spatial_size, batch_c, D, 5)

        return x
