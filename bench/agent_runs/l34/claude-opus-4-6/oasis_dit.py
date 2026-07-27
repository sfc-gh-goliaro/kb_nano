"""Optimized Oasis diffusion transformer with flash attention and fused adaLN."""

from __future__ import annotations

import math
from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.oasis_rotary import OasisRotaryEmbedding, oasis_apply_rotary_emb
from fastkernels.tasks.baseline.L2.oasis_final_layer import OasisFinalLayer
from fastkernels.tasks.baseline.L2.oasis_patch_embed import OasisPatchEmbed
from fastkernels.tasks.baseline.L2.oasis_timestep_embedder import OasisTimestepEmbedder


# ---------- Triton fused adaLN kernel ----------

@triton.jit
def _fused_adaln_fwd(
    X_ptr, Out_ptr, Shift_ptr, Scale_ptr,
    stride_x_row, stride_shift_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_off = row * stride_x_row + cols
    s_off = row * stride_shift_row + cols

    x = tl.load(X_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = xc * rstd

    shift = tl.load(Shift_ptr + s_off, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(Scale_ptr + s_off, mask=mask, other=0.0).to(tl.float32)

    out = xn * (1.0 + scale) + shift
    tl.store(Out_ptr + x_off, out, mask=mask)


def fused_adaln(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused LayerNorm + adaptive modulation: norm(x) * (1 + scale) + shift."""
    orig_shape = x.shape
    N = x.shape[-1]
    BLOCK_N = triton.next_power_of_2(N)

    if BLOCK_N > 8192 or x.dtype == torch.float32:
        x_norm = F.layer_norm(x.float(), (N,), eps=eps)
        while shift.dim() < x.dim():
            shift = shift.unsqueeze(-2)
            scale = scale.unsqueeze(-2)
        return (x_norm * (1 + scale.float()) + shift.float()).to(x.dtype)

    x_flat = x.contiguous().reshape(-1, N)
    num_rows = x_flat.shape[0]

    shift_exp = shift
    scale_exp = scale
    while shift_exp.dim() < len(orig_shape):
        shift_exp = shift_exp.unsqueeze(-2)
        scale_exp = scale_exp.unsqueeze(-2)
    shift_flat = shift_exp.expand(orig_shape).contiguous().reshape(-1, N)
    scale_flat = scale_exp.expand(orig_shape).contiguous().reshape(-1, N)

    out = torch.empty_like(x_flat)
    _fused_adaln_fwd[(num_rows,)](
        x_flat, out, shift_flat, scale_flat,
        stride_x_row=N, stride_shift_row=N,
        N=N, eps=eps, BLOCK_N=BLOCK_N,
    )
    return out.reshape(orig_shape)


# ---------- Optimized attention forward functions ----------

def spatial_attn_forward(
    x: torch.Tensor,
    to_qkv_weight: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
    rotary_emb: OasisRotaryEmbedding,
    num_heads: int,
) -> torch.Tensor:
    bsz, time, height, width, D = x.shape
    bt = bsz * time
    hw = height * width
    dim_head = D // num_heads

    qkv = F.linear(x.reshape(bt * hw, D), to_qkv_weight)
    qkv = qkv.reshape(bt, height, width, 3 * num_heads * dim_head)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(bt, height, width, num_heads, dim_head).permute(0, 3, 1, 2, 4)
    k = k.reshape(bt, height, width, num_heads, dim_head).permute(0, 3, 1, 2, 4)

    freqs = rotary_emb.get_axial_freqs(height, width)
    q = oasis_apply_rotary_emb(freqs, q)
    k = oasis_apply_rotary_emb(freqs, k)

    q = q.reshape(bt, num_heads, hw, dim_head)
    k = k.reshape(bt, num_heads, hw, dim_head)
    v = v.reshape(bt, hw, num_heads, dim_head)

    if HAS_FLASH_ATTN and x.dtype != torch.float32:
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_c = v.contiguous()
        out = flash_attn_func(q_fa, k_fa, v_c, causal=False)
    else:
        v_sdpa = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v_sdpa, is_causal=False)
        out = out.transpose(1, 2)

    out = out.reshape(bsz, time, height, width, num_heads * dim_head)
    return F.linear(out, to_out_weight, to_out_bias)


def temporal_attn_forward(
    x: torch.Tensor,
    to_qkv_weight: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
    rotary_emb: OasisRotaryEmbedding,
    num_heads: int,
    is_causal: bool,
) -> torch.Tensor:
    bsz, time, height, width, D = x.shape
    bhw = bsz * height * width
    dim_head = D // num_heads

    x_t = x.permute(0, 2, 3, 1, 4).reshape(bhw, time, D)
    qkv = F.linear(x_t, to_qkv_weight)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(bhw, time, num_heads, dim_head).transpose(1, 2)
    k = k.reshape(bhw, time, num_heads, dim_head).transpose(1, 2)
    v = v.reshape(bhw, time, num_heads, dim_head)

    q = rotary_emb.rotate_queries_or_keys(q, rotary_emb.freqs)
    k = rotary_emb.rotate_queries_or_keys(k, rotary_emb.freqs)

    if HAS_FLASH_ATTN and x.dtype != torch.float32:
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_c = v.contiguous()
        out = flash_attn_func(q_fa, k_fa, v_c, causal=is_causal)
    else:
        v_sdpa = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v_sdpa, is_causal=is_causal)
        out = out.transpose(1, 2)

    out = out.reshape(bsz, height, width, time, num_heads * dim_head)
    out = out.permute(0, 3, 1, 2, 4)
    return F.linear(out, to_out_weight, to_out_bias)


# ---------- Optimized block ----------

def _modulate_expand(mod_tensor: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Repeat mod along batch dim to match x, used when bsz*time != bsz."""
    fixed_dims = [1] * len(mod_tensor.shape[1:])
    return mod_tensor.repeat(x.shape[0] // mod_tensor.shape[0], *fixed_dims)


def _apply_gate(x_attn: torch.Tensor, gate: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    fixed_dims = [1] * len(gate.shape[1:])
    g = gate.repeat(x_ref.shape[0] // gate.shape[0], *fixed_dims)
    while g.dim() < x_attn.dim():
        g = g.unsqueeze(-2)
    return g * x_attn


class OptimizedSpatioTemporalDiTBlock(nn.Module):
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
        from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
        from fastkernels.tasks.baseline.L1.silu import SiLU
        from fastkernels.tasks.baseline.L2.oasis_mlp import OasisMLP
        from fastkernels.tasks.baseline.L2.oasis_spatial_axial_attention import OasisSpatialAxialAttention
        from fastkernels.tasks.baseline.L2.oasis_temporal_axial_attention import OasisTemporalAxialAttention

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.s_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_attn = OasisSpatialAxialAttention(
            hidden_size, heads=num_heads, dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_mlp = OasisMLP(hidden_size, hidden_features=int(hidden_size * mlp_ratio), approximate_tanh=True)
        self.s_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.t_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_attn = OasisTemporalAxialAttention(
            hidden_size, heads=num_heads, dim_head=hidden_size // num_heads,
            rotary_emb=temporal_rotary_emb, is_causal=is_causal,
        )
        self.t_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_mlp = OasisMLP(hidden_size, hidden_features=int(hidden_size * mlp_ratio), approximate_tanh=True)
        self.t_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Spatial path
        s_mod = F.linear(F.silu(c), self.s_adaLN_modulation[1].weight, self.s_adaLN_modulation[1].bias)
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = s_mod.chunk(6, dim=-1)

        s_shift = _modulate_expand(s_shift_msa, x)
        s_scale = _modulate_expand(s_scale_msa, x)
        x_mod = fused_adaln(x, s_shift, s_scale)
        sa_out = spatial_attn_forward(
            x_mod, self.s_attn.to_qkv.weight, self.s_attn.to_out.weight, self.s_attn.to_out.bias,
            self.s_attn.rotary_emb, self.num_heads,
        )
        x = x + _apply_gate(sa_out, s_gate_msa, x)

        s_shift2 = _modulate_expand(s_shift_mlp, x)
        s_scale2 = _modulate_expand(s_scale_mlp, x)
        x_mod = fused_adaln(x, s_shift2, s_scale2)
        mlp_out = F.linear(
            F.gelu(F.linear(x_mod, self.s_mlp.fc1.weight, self.s_mlp.fc1.bias), approximate="tanh"),
            self.s_mlp.fc2.weight, self.s_mlp.fc2.bias,
        )
        x = x + _apply_gate(mlp_out, s_gate_mlp, x)

        # Temporal path
        t_mod = F.linear(F.silu(c), self.t_adaLN_modulation[1].weight, self.t_adaLN_modulation[1].bias)
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = t_mod.chunk(6, dim=-1)

        t_shift = _modulate_expand(t_shift_msa, x)
        t_scale = _modulate_expand(t_scale_msa, x)
        x_mod = fused_adaln(x, t_shift, t_scale)
        ta_out = temporal_attn_forward(
            x_mod, self.t_attn.to_qkv.weight, self.t_attn.to_out.weight, self.t_attn.to_out.bias,
            self.t_attn.rotary_emb, self.num_heads, self.t_attn.is_causal,
        )
        x = x + _apply_gate(ta_out, t_gate_msa, x)

        t_shift2 = _modulate_expand(t_shift_mlp, x)
        t_scale2 = _modulate_expand(t_scale_mlp, x)
        x_mod = fused_adaln(x, t_shift2, t_scale2)
        mlp_out = F.linear(
            F.gelu(F.linear(x_mod, self.t_mlp.fc1.weight, self.t_mlp.fc1.bias), approximate="tanh"),
            self.t_mlp.fc2.weight, self.t_mlp.fc2.bias,
        )
        x = x + _apply_gate(mlp_out, t_gate_mlp, x)

        return x


# ---------- Main model ----------

class OasisDiT(nn.Module):
    def __init__(
        self,
        *,
        input_h: int = 18,
        input_w: int = 32,
        patch_size: int = 2,
        in_channels: int = 16,
        hidden_size: int = 1024,
        depth: int = 16,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        external_cond_dim: int = 25,
        max_frames: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames
        self.hidden_size = hidden_size

        self.x_embedder = OasisPatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = OasisTimestepEmbedder(hidden_size)
        head_dim = hidden_size // num_heads
        self.spatial_rotary_emb = OasisRotaryEmbedding(dim=head_dim // 2, freqs_for="pixel", max_freq=256)
        self.temporal_rotary_emb = OasisRotaryEmbedding(dim=head_dim, freqs_for="lang")
        self.external_cond = Linear(external_cond_dim, hidden_size, bias=True) if external_cond_dim > 0 else nn.Identity()
        self.blocks = nn.ModuleList([
            OptimizedSpatioTemporalDiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, is_causal=True,
                spatial_rotary_emb=self.spatial_rotary_emb, temporal_rotary_emb=self.temporal_rotary_emb,
            )
            for _ in range(depth)
        ])
        self.final_layer = OasisFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, (Linear, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        if self.final_layer.linear.bias is not None:
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, external_cond: torch.Tensor | None = None) -> torch.Tensor:
        bsz, time, channels, height, width = x.shape

        # Patch embedding
        x = x.reshape(bsz * time, channels, height, width)
        x = self.x_embedder(x)
        x = x.reshape(bsz, time, x.shape[1], x.shape[2], x.shape[3])

        # Timestep embedding
        t = t.reshape(bsz * time)
        c = self.t_embedder(t).reshape(bsz, time, -1)
        if torch.is_tensor(external_cond):
            c = c + F.linear(external_cond, self.external_cond.weight, self.external_cond.bias)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final layer
        final_mod = c
        for layer in self.final_layer.adaLN_modulation:
            final_mod = layer(final_mod)
        shift, scale = final_mod.chunk(2, dim=-1)
        shift_e = _modulate_expand(shift, x)
        scale_e = _modulate_expand(scale, x)
        x = fused_adaln(x, shift_e, scale_e)
        x = F.linear(x, self.final_layer.linear.weight, self.final_layer.linear.bias)

        # Unpatchify
        x = x.reshape(bsz * time, x.shape[2], x.shape[3], x.shape[4])
        x = self.unpatchify(x)
        return x.reshape(bsz, time, x.shape[1], x.shape[2], x.shape[3])
