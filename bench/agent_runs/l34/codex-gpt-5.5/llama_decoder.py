from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.llama_mlp import LlamaMLP


if triton is not None:
    @triton.jit
    def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(x * x, axis=0)
        rstd = tl.rsqrt(ss / n_cols + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(y_ptr + row * n_cols + offs, y, mask=mask)


    @triton.jit
    def _add_rms_norm_kernel(x_ptr, r_ptr, w_ptr, y_ptr, ro_ptr, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        v = x + r
        tl.store(ro_ptr + row * n_cols + offs, v, mask=mask)
        ss = tl.sum(v * v, axis=0)
        rstd = tl.rsqrt(ss / n_cols + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = v * rstd * w
        tl.store(y_ptr + row * n_cols + offs, y, mask=mask)


def _num_warps(block: int) -> int:
    if block >= 8192:
        return 8
    if block >= 2048:
        return 4
    return 1


class _FastRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.eps = eps
        self.hidden_size = hidden_size

    def _torch_norm(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        y = xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + self.eps)
        y = y * self.weight.float()
        return y.to(orig_dtype)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            if triton is None or not hidden_states.is_cuda or not hidden_states.is_contiguous():
                return self._torch_norm(hidden_states)
            n_cols = hidden_states.shape[-1]
            y = torch.empty_like(hidden_states)
            rows = hidden_states.numel() // n_cols
            block = triton.next_power_of_2(n_cols)
            _rms_norm_kernel[(rows,)](
                hidden_states, self.weight, y, n_cols, self.eps, BLOCK=block,
                num_warps=_num_warps(block),
            )
            return y

        if (
            triton is None
            or not hidden_states.is_cuda
            or not residual.is_cuda
            or not hidden_states.is_contiguous()
            or not residual.is_contiguous()
        ):
            residual_out = hidden_states + residual
            return self._torch_norm(residual_out), residual_out

        n_cols = hidden_states.shape[-1]
        y = torch.empty_like(hidden_states)
        residual_out = torch.empty_like(hidden_states)
        rows = hidden_states.numel() // n_cols
        block = triton.next_power_of_2(n_cols)
        _add_rms_norm_kernel[(rows,)](
            hidden_states, residual, self.weight, y, residual_out, n_cols, self.eps, BLOCK=block,
            num_warps=_num_warps(block),
        )
        return y, residual_out


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        rotary_emb: nn.Module | None = None,
        bias: bool = False,
        qk_norm: bool = False,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            rotary_emb=rotary_emb,
            bias=bias,
            qk_norm=qk_norm,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
        )
        self.mlp = LlamaMLP(config, quant_config=quant_config)
        self.input_layernorm = _FastRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _FastRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
