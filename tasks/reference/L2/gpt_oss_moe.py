"""Packed MXFP4 Triton reference for gpt_oss_moe.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.

GPT-OSS production weights are packed MXFP4. This reference keeps that packed
format and delegates routing/fused expert matmuls to the L1 ``mxfp4_moe``
Triton reference instead of dequantizing expert weights in Python.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kb_nano.infra.tp import _tp_rank, _tp_size
from kb_nano.tasks.reference.L1.linear import Linear
from kb_nano.tasks.reference.L1.allreduce import AllReduce
from kb_nano.tasks.reference.L1.mxfp4_moe import Mxfp4MoE


def _round_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


class GptOssMoE(nn.Module):
    MXFP4_BLOCK = 32

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.intermediate_size // tp
        self.router = Linear(config.hidden_size, config.num_local_experts, bias=True)
        e = config.num_local_experts
        i_pad = _round_up(self.intermediate_per_tp, 64)
        h = config.hidden_size
        blk = self.MXFP4_BLOCK
        self._I_pad = i_pad
        self.w13_weight = nn.Parameter(
            torch.zeros(e, 2 * i_pad, h // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w13_weight_scale = nn.Parameter(
            torch.zeros(e, 2 * i_pad, h // blk, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w13_bias = nn.Parameter(
            torch.zeros(e, 2 * i_pad, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(e, h, i_pad // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w2_weight_scale = nn.Parameter(
            torch.zeros(e, h, i_pad // blk, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w2_bias = nn.Parameter(
            torch.zeros(e, h, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.w13_weight.weight_loader = self._w13_weight_loader
        self.w13_weight_scale.weight_loader = self._w13_scale_loader
        self.w13_bias.weight_loader = self._w13_bias_loader
        self.w2_weight.weight_loader = self._w2_weight_loader
        self.w2_weight_scale.weight_loader = self._w2_scale_loader
        self.w2_bias.weight_loader = self._w2_bias_loader
        self.allreduce = AllReduce()
        self.mxfp4_moe = Mxfp4MoE()
        self._quant_config = None
        self._processed = False
        self._use_custom_op = False
        self._layer_name = ""

    def _w13_weight_loader(self, param, loaded_weight):
        if loaded_weight.ndim == 4:
            e, n, nb, bs = loaded_weight.shape
            loaded_weight = loaded_weight.reshape(e, n, nb * bs)
        rank = _tp_rank()
        i = self.intermediate_per_tp
        start = 2 * rank * i
        param.data[:, :2 * i, :].copy_(loaded_weight[:, start:start + 2 * i, :])

    def _w13_scale_loader(self, param, loaded_weight):
        rank = _tp_rank()
        i = self.intermediate_per_tp
        start = 2 * rank * i
        param.data[:, :2 * i, :].copy_(loaded_weight[:, start:start + 2 * i, :])

    def _w13_bias_loader(self, param, loaded_weight):
        rank = _tp_rank()
        i = self.intermediate_per_tp
        start = 2 * rank * i
        param.data[:, :2 * i].copy_(loaded_weight[:, start:start + 2 * i])

    def _w2_weight_loader(self, param, loaded_weight):
        if loaded_weight.ndim == 4:
            e, h, nb, bs = loaded_weight.shape
            loaded_weight = loaded_weight.reshape(e, h, nb * bs)
        rank = _tp_rank()
        i_half = self.intermediate_per_tp // 2
        param.data[:, :, :i_half].copy_(
            loaded_weight[:, :, rank * i_half:rank * i_half + i_half],
        )

    def _w2_scale_loader(self, param, loaded_weight):
        rank = _tp_rank()
        i_blk = self.intermediate_per_tp // self.MXFP4_BLOCK
        param.data[:, :, :i_blk].copy_(
            loaded_weight[:, :, rank * i_blk:rank * i_blk + i_blk],
        )

    def _w2_bias_loader(self, param, loaded_weight):
        if _tp_rank() == 0:
            param.data.copy_(loaded_weight)
        else:
            param.data.zero_()

    def process_weights_after_loading(self):
        if self._processed:
            return
        self.w13_bias.data = self.w13_bias.data.float()
        self.w2_bias.data = self.w2_bias.data.float()
        self._w13_swizzled, w13_precision = Mxfp4MoE.prepare_weight(
            self.w13_weight.data, self.w13_weight_scale.data,
        )
        self._w2_swizzled, w2_precision = Mxfp4MoE.prepare_weight(
            self.w2_weight.data, self.w2_weight_scale.data,
        )
        del self.w13_weight, self.w2_weight
        del self.w13_weight_scale, self.w2_weight_scale
        self._quant_config = Mxfp4MoE.make_quant_config(
            w1_precision=w13_precision,
            w2_precision=w2_precision,
            w1_bias=self.w13_bias.data,
            w2_bias=self.w2_bias.data,
        )
        self._processed = True

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._processed:
            self.process_weights_after_loading()
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = self.router(hidden_states)
        output = self.mxfp4_moe(
            hidden_states=hidden_states,
            w1=self._w13_swizzled,
            w2=self._w2_swizzled,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=True,
            quant_config=self._quant_config,
            apply_router_weight_on_input=False,
        )
        if self.tp_size > 1:
            output = self.allreduce(output)
        return output.view(orig_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.forward_impl(hidden_states)
