import torch
import torch.nn as nn
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.gpt_oss_moe import GptOssMoE


@triton.jit
def _rms_norm_kernel(
    X_ptr, Out_ptr, W_ptr,
    stride_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * stride_row
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.math.rsqrt(tl.sum(x * x, axis=0) * (1.0 / N) + eps)
    tl.store(Out_ptr + base + cols, (x * rrms * w).to(Out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_add_rms_norm_kernel(
    X_ptr, Res_ptr, W_ptr,
    stride_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * stride_row
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Res_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    s = x + r
    tl.store(Res_ptr + base + cols, s.to(Res_ptr.dtype.element_ty), mask=mask)
    rrms = tl.math.rsqrt(tl.sum(s * s, axis=0) * (1.0 / N) + eps)
    tl.store(X_ptr + base + cols, (s * rrms * w).to(X_ptr.dtype.element_ty), mask=mask)


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            bias=True,
            o_proj_bias=True,
            use_sinks=True,
            sliding_window=config.sliding_window,
            layer_idx=layer_idx,
        )
        self.mlp = GptOssMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self._eps = config.rms_norm_eps
        self._H = config.hidden_size
        self._BN = triton.next_power_of_2(config.hidden_size)
        self._nw = 4 if config.hidden_size <= 1024 else 8

    def forward(self, positions, hidden_states, residual, rotary_emb):
        H = self._H
        BN = self._BN
        eps = self._eps
        nw = self._nw
        in_w = self.input_layernorm.weight
        post_w = self.post_attention_layernorm.weight

        n = hidden_states.shape[0]
        if residual is None:
            residual = hidden_states
            hidden_states = torch.empty_like(hidden_states)
            _rms_norm_kernel[(n,)](
                residual, hidden_states, in_w, H, H, eps, BN, num_warps=nw
            )
        else:
            _fused_add_rms_norm_kernel[(n,)](
                hidden_states, residual, in_w, H, H, eps, BN, num_warps=nw
            )

        hidden_states = self.self_attn(positions, hidden_states, rotary_emb=rotary_emb)

        _fused_add_rms_norm_kernel[(hidden_states.shape[0],)](
            hidden_states, residual, post_w, H, H, eps, BN, num_warps=nw
        )

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
