from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - fallback path for non-Triton environments
    triton = None
    tl = None

from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.oasis_rotary import OasisRotaryEmbedding
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L2.oasis_mlp import OasisMLP
from fastkernels.tasks.baseline.L2.oasis_spatial_axial_attention import OasisSpatialAxialAttention
from fastkernels.tasks.baseline.L2.oasis_temporal_axial_attention import OasisTemporalAxialAttention


if triton is not None:

    @triton.jit
    def _ln_mod_kernel(
        x_ptr,
        shift_ptr,
        scale_ptr,
        y_ptr,
        n_rows: tl.constexpr,
        hidden: tl.constexpr,
        rows_per_first: tl.constexpr,
        param_rows: tl.constexpr,
        shift_s0: tl.constexpr,
        shift_s1: tl.constexpr,
        scale_s0: tl.constexpr,
        scale_s1: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < hidden
        x = tl.load(x_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / hidden
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / hidden
        inv_std = tl.rsqrt(var + eps)
        p_row = (row // rows_per_first) % param_rows
        shift = tl.load(shift_ptr + p_row * shift_s0 + offs * shift_s1, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + p_row * scale_s0 + offs * scale_s1, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std * (1.0 + scale) + shift
        tl.store(y_ptr + row * hidden + offs, y, mask=mask)

    @triton.jit
    def _gate_add_kernel(
        residual_ptr,
        x_ptr,
        gate_ptr,
        y_ptr,
        total: tl.constexpr,
        hidden: tl.constexpr,
        rows_per_first: tl.constexpr,
        gate_rows: tl.constexpr,
        gate_s0: tl.constexpr,
        gate_s1: tl.constexpr,
        block: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * block + tl.arange(0, block)
        mask = offs < total
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        col = offs % hidden
        row = offs // hidden
        p_row = (row // rows_per_first) % gate_rows
        gate = tl.load(gate_ptr + p_row * gate_s0 + col * gate_s1, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + offs, residual + vals * gate, mask=mask)


def _broadcast_param(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if p.shape[0] != x.shape[0] and p.shape[0] != 1:
        fixed_dims = [1] * len(p.shape[1:])
        p = p.repeat(x.shape[0] // p.shape[0], *fixed_dims)
    while p.dim() < x.dim():
        p = p.unsqueeze(-2)
    return p


def _modulate_fallback(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    shift = _broadcast_param(shift, x)
    scale = _broadcast_param(scale, x)
    return x * (1.0 + scale) + shift


def _gate_fallback(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return _broadcast_param(g, x) * x


def _triton_block_size(hidden: int) -> int:
    return triton.next_power_of_2(hidden)


def _triton_warps(block: int) -> int:
    if block >= 4096:
        return 8
    if block >= 2048:
        return 8
    if block >= 1024:
        return 4
    return 4


def _can_use_triton_2d_param(x: torch.Tensor, p: torch.Tensor) -> bool:
    return (
        triton is not None
        and x.is_cuda
        and p.is_cuda
        and x.is_contiguous()
        and p.dim() == 2
        and x.dim() >= 2
        and x.shape[-1] == p.shape[-1]
        and x.shape[0] % p.shape[0] == 0
        and x.shape[-1] <= 8192
    )


def _norm_modulate(
    norm: nn.Module,
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if _can_use_triton_2d_param(x, shift) and _can_use_triton_2d_param(x, scale):
        hidden = x.shape[-1]
        block = _triton_block_size(hidden)
        y = torch.empty_like(x, memory_format=torch.contiguous_format)
        n_rows = x.numel() // hidden
        rows_per_first = n_rows // x.shape[0]
        _ln_mod_kernel[(n_rows,)](
            x,
            shift,
            scale,
            y,
            n_rows,
            hidden,
            rows_per_first,
            shift.shape[0],
            shift.stride(0),
            shift.stride(1),
            scale.stride(0),
            scale.stride(1),
            float(getattr(norm, "eps", 1e-6)),
            block,
            num_warps=_triton_warps(block),
        )
        return y
    return _modulate_fallback(norm(x), shift, scale)


def _gate_add(residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if (
        _can_use_triton_2d_param(x, gate)
        and residual.is_cuda
        and residual.is_contiguous()
        and residual.shape == x.shape
        and residual.dtype == x.dtype
    ):
        hidden = x.shape[-1]
        total = x.numel()
        rows_per_first = (total // hidden) // x.shape[0]
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        block = 256
        _gate_add_kernel[(triton.cdiv(total, block),)](
            residual,
            x,
            gate,
            out,
            total,
            hidden,
            rows_per_first,
            gate.shape[0],
            gate.stride(0),
            gate.stride(1),
            block,
            num_warps=4,
        )
        return out
    return residual + _gate_fallback(x, gate)


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
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = (
            self.s_adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = _gate_add(x, self.s_attn(_norm_modulate(self.s_norm1, x, s_shift_msa, s_scale_msa)), s_gate_msa)
        x = _gate_add(x, self.s_mlp(_norm_modulate(self.s_norm2, x, s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = (
            self.t_adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = _gate_add(x, self.t_attn(_norm_modulate(self.t_norm1, x, t_shift_msa, t_scale_msa)), t_gate_msa)
        x = _gate_add(x, self.t_mlp(_norm_modulate(self.t_norm2, x, t_shift_mlp, t_scale_mlp)), t_gate_mlp)
        return x
