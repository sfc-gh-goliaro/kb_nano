from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L2.ada_layer_norm import AdaLayerNormZeroSingle
from fastkernels.tasks.baseline.L2.flux_attention import FluxAttention
from fastkernels.tasks.baseline.L2.parallel_linear import ReplicatedLinear


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
        residual = torch.cat((encoder_hidden_states, hidden_states), dim=1)

        norm_hidden_states, gate = self.norm(residual, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        if joint_attention_kwargs is None:
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )
        else:
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )

        hidden_states = self.proj_out(torch.cat((attn_output, mlp_hidden_states), dim=2))
        hidden_states = torch.addcmul(
            residual, hidden_states, gate.unsqueeze(1), value=1.0
        )

        if hidden_states.dtype is torch.float16:
            hidden_states = hidden_states.clamp(-65504, 65504)

        return hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
