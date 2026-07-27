from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L2.ada_layer_norm import AdaLayerNormZeroSingle
from fastkernels.tasks.baseline.L2.flux_attention import FluxAttention
from fastkernels.tasks.baseline.L2.parallel_linear import ReplicatedLinear


@triton.jit
def _fused_gate_res_clip_kernel(
    out_ptr, proj_ptr, gate_ptr, res_ptr,
    S, D,
    IS_FP16: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b_idx = tl.program_id(0)
    s_idx = tl.program_id(1)
    d_block = tl.program_id(2)

    d_offs = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_offs < D

    row = b_idx * S + s_idx
    base = row * D

    p = tl.load(proj_ptr + base + d_offs, mask=mask)
    g = tl.load(gate_ptr + b_idx * D + d_offs, mask=mask)
    r = tl.load(res_ptr + base + d_offs, mask=mask)

    result = g * p + r
    if IS_FP16:
        result = tl.minimum(tl.maximum(result, -65504.0), 65504.0)

    tl.store(out_ptr + base + d_offs, result, mask=mask)


class FluxSingleTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = ReplicatedLinear(
            dim, self.mlp_hidden_dim, bias=True, quant_config=quant_config
        )
        self.act_mlp = GELU(approximate="tanh")
        self.proj_out = ReplicatedLinear(
            dim + self.mlp_hidden_dim, dim, bias=True, quant_config=quant_config
        )

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = self.proj_out(hidden_states)

        gate = gate.contiguous()
        B, S, D = hidden_states.shape
        output = torch.empty_like(hidden_states)
        BLOCK_D = min(triton.next_power_of_2(D), 1024)
        grid = (B, S, triton.cdiv(D, BLOCK_D))
        _fused_gate_res_clip_kernel[grid](
            output, hidden_states, gate, residual,
            S, D,
            IS_FP16=(hidden_states.dtype == torch.float16),
            BLOCK_D=BLOCK_D,
        )

        encoder_hidden_states, hidden_states = (
            output[:, :text_seq_len],
            output[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states
