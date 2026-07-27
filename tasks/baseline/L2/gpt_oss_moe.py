"""GPT-OSS MoE: MXFP4-native fused MoE composed from FastKernels L1 ops.

32 experts (top-4, softmax routing), router bias, expert gate/up/down biases,
OAI SwiGLU activation fused inside the Triton matmul_ogs kernel.

Expert weights are kept in packed MXFP4 uint8 format (2× FP4 per byte) with
E8M0 block scales, matching vLLM's Mxfp4MoEMethod Triton backend exactly.
No dequantization is performed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.linear import Linear
from ..L1.mxfp4_moe import Mxfp4MoE


def _round_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


class GptOssMoE(nn.Module):
    """MXFP4-native MoE composed from FastKernels L1 ops.

    Weights stay in packed uint8 MXFP4 format. Routing, swizzling and the
    fused matmul_ogs forward are all delegated to ``L1.mxfp4_moe``.
    """

    MXFP4_BLOCK = 32
    # Upper bound on tokens per MoE call, used to pick the FlashInfer tile config once.
    _FI_TUNE_MAX_TOKENS = 8192

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.intermediate_size // tp

        self.router = Linear(config.hidden_size, config.num_local_experts, bias=True)

        E = config.num_local_experts
        # The FlashInfer TRTLLM-gen MoE wants both the intermediate and the hidden axis
        # aligned to 256; the Triton path only needs the intermediate aligned to 64 and
        # leaves hidden alone. gpt-oss-20b TP1: 2880 -> 3072 on both axes; 120b TP2:
        # I 1440 -> 1536, H 2880 -> 3072. Zero-init padding is inert: it survives the
        # kernel's permutation as zeros, and an E8M0 scale byte of 0 times FP4 0 is 0.
        self._use_flashinfer = Mxfp4MoE.sm100_flashinfer_available()
        align = 256 if self._use_flashinfer else 64
        I_pad = _round_up(self.intermediate_per_tp, align)
        H = _round_up(config.hidden_size, align)
        BLK = self.MXFP4_BLOCK

        self._I_pad = I_pad
        self._H_pad = H

        # Expert weights in packed MXFP4 uint8 (2× FP4 per byte)
        self.w13_weight = nn.Parameter(
            torch.zeros(E, 2 * I_pad, H // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w13_weight_scale = nn.Parameter(
            torch.zeros(E, 2 * I_pad, H // BLK, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w13_bias = nn.Parameter(
            torch.zeros(E, 2 * I_pad, dtype=torch.bfloat16),
            requires_grad=False,
        )

        self.w2_weight = nn.Parameter(
            torch.zeros(E, H, I_pad // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w2_weight_scale = nn.Parameter(
            torch.zeros(E, H, I_pad // BLK, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w2_bias = nn.Parameter(
            torch.zeros(E, H, dtype=torch.bfloat16),
            requires_grad=False,
        )

        # Set up weight loaders for checkpoint loading
        self.w13_weight.weight_loader = self._w13_weight_loader
        self.w13_weight_scale.weight_loader = self._w13_scale_loader
        self.w13_bias.weight_loader = self._w13_bias_loader
        self.w2_weight.weight_loader = self._w2_weight_loader
        self.w2_weight_scale.weight_loader = self._w2_scale_loader
        self.w2_bias.weight_loader = self._w2_bias_loader

        self.allreduce = AllReduce()
        self.mxfp4_moe = Mxfp4MoE()

        # Populated after process_weights_after_loading
        self._quant_config = None
        self._processed = False

        # Custom-op dispatch for torch.compile (set by engine after model init)
        self._use_custom_op = False
        self._layer_name = ""

    def _w13_weight_loader(self, param, loaded_weight):
        """Load w13 MXFP4 packed weight with TP sharding.

        Checkpoint shape: [E, 2*I_full, num_blocks, 16] (4D blocks) or
                          [E, 2*I_full, H//2] (pre-flattened).
        Gate/up rows are interleaved (gate_0, up_0, gate_1, up_1, ...);
        we keep them interleaved, matching vLLM's Triton kernel expectation.
        """
        if loaded_weight.ndim == 4:
            E, N, nb, bs = loaded_weight.shape
            loaded_weight = loaded_weight.reshape(E, N, nb * bs)
        rank = _tp_rank()
        I = self.intermediate_per_tp
        start = 2 * rank * I
        # Bound the hidden axis as well: it is padded on the FlashInfer path.
        cols = loaded_weight.shape[-1]
        param.data[:, :2*I, :cols].copy_(loaded_weight[:, start : start + 2*I, :])

    def _w13_scale_loader(self, param, loaded_weight):
        """Load w13 scales with TP shard, keeping interleaved layout."""
        rank = _tp_rank()
        I = self.intermediate_per_tp
        start = 2 * rank * I
        cols = loaded_weight.shape[-1]
        param.data[:, :2*I, :cols].copy_(loaded_weight[:, start : start + 2*I, :])

    def _w13_bias_loader(self, param, loaded_weight):
        """Load w13 bias [E, 2*I] with TP shard, keeping interleaved layout."""
        rank = _tp_rank()
        I = self.intermediate_per_tp
        start = 2 * rank * I
        param.data[:, :2*I].copy_(loaded_weight[:, start : start + 2*I])

    def _w2_weight_loader(self, param, loaded_weight):
        """Load w2 MXFP4 packed weight with TP shard.

        Checkpoint shape: [E, H, num_blocks, 16] (4D blocks) or
                          [E, H, I//2] (pre-flattened).
        """
        if loaded_weight.ndim == 4:
            E, H, nb, bs = loaded_weight.shape
            loaded_weight = loaded_weight.reshape(E, H, nb * bs)
        tp, rank = _tp_size(), _tp_rank()
        I_half = self.intermediate_per_tp // 2
        rows = loaded_weight.shape[1]
        param.data[:, :rows, :I_half].copy_(
            loaded_weight[:, :, rank * I_half : rank * I_half + I_half]
        )

    def _w2_scale_loader(self, param, loaded_weight):
        """Load w2 scales with TP shard."""
        tp, rank = _tp_size(), _tp_rank()
        I_blk = self.intermediate_per_tp // self.MXFP4_BLOCK
        rows = loaded_weight.shape[1]
        param.data[:, :rows, :I_blk].copy_(
            loaded_weight[:, :, rank * I_blk : rank * I_blk + I_blk]
        )

    def _w2_bias_loader(self, param, loaded_weight):
        """Load w2 bias [E, H]. Only rank 0 loads; others zero (reduced by allreduce)."""
        if _tp_rank() == 0:
            param.data[:, :loaded_weight.shape[1]].copy_(loaded_weight)
        else:
            param.data.zero_()

    def process_weights_after_loading(self):
        """Swizzle MXFP4 weights for Triton matmul_ogs and build quant config.

        Must be called after all weights are loaded and moved to GPU.
        """
        if self._processed:
            return

        # Biases must be float32 for either kernel
        self.w13_bias.data = self.w13_bias.data.float()
        self.w2_bias.data = self.w2_bias.data.float()

        if self._use_flashinfer:
            # Blackwell: lay the weights out for trtllm_fp4_block_scale_moe, which is
            # what vLLM itself runs on sm100. Replace the tensors in place rather than
            # deleting the parameters -- the scales have to stay live, and a Parameter
            # cannot hold float8 cleanly through our dtype-casting loader, so keep the
            # fp8 view alongside.
            FW13, FS13, FB13, FW2, FS2, FB2 = Mxfp4MoE.prepare_weight_flashinfer(
                self.w13_weight.data, self.w13_weight_scale.data, self.w13_bias.data,
                self.w2_weight.data, self.w2_weight_scale.data, self.w2_bias.data,
            )
            self.w13_weight.data = FW13
            self.w2_weight.data = FW2
            self.w13_weight_scale.data = FS13.view(torch.uint8)
            self.w2_weight_scale.data = FS2.view(torch.uint8)
            self._fi_w13_scale = FS13
            self._fi_w2_scale = FS2
            self.w13_bias.data = FB13
            self.w2_bias.data = FB2
            E = self.num_experts
            dev = FW13.device
            # OAI SwiGLU as the kernel parameterises it: gate upper-clamped, up
            # two-sided-clamped, s*(up+1). The gate/up pair order is handled by the
            # row swap inside prepare_weight_flashinfer.
            self._fi_alpha = torch.full((E,), 1.702, dtype=torch.float32, device=dev)
            self._fi_beta = torch.full((E,), 1.0, dtype=torch.float32, device=dev)
            self._fi_clamp = torch.full((E,), 7.0, dtype=torch.float32, device=dev)
            self._processed = True
            return

        w13_weight, w13_precision = Mxfp4MoE.prepare_weight(
            self.w13_weight.data, self.w13_weight_scale.data
        )
        w2_weight, w2_precision = Mxfp4MoE.prepare_weight(
            self.w2_weight.data, self.w2_weight_scale.data
        )

        # prepare_weight returns triton_kernels.Tensor objects, not
        # torch.Tensor; store as plain attributes (the original nn.Parameters
        # are no longer used)
        del self.w13_weight, self.w2_weight
        del self.w13_weight_scale, self.w2_weight_scale
        self._w13_swizzled = w13_weight
        self._w2_swizzled = w2_weight

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

        if self._use_flashinfer:
            h = hidden_states
            if self._H_pad != self.hidden_size:
                # Required, not optional: the kernel derives weight_scale_vec_size as
                # hidden / scale_columns and rejects anything but 16 or 32, so an
                # unpadded 2880 against 3072/32=96 scale columns gives 30 and fails
                # ("unsupported weight_scale_vec_size"). vLLM's call site passes
                # `x_quant = x` unpadded, which means it pads the hidden dim model-wide
                # instead -- so our local pad is equivalent, not a divergence.
                h = F.pad(h, (0, self._H_pad - self.hidden_size))
            output = Mxfp4MoE.forward_flashinfer(
                h, router_logits,
                w13=self.w13_weight.data, w13_scale=self._fi_w13_scale,
                w13_bias=self.w13_bias.data,
                w2=self.w2_weight.data, w2_scale=self._fi_w2_scale,
                w2_bias=self.w2_bias.data,
                alpha=self._fi_alpha, beta=self._fi_beta,
                clamp_limit=self._fi_clamp,
                num_experts=self.num_experts, top_k=self.top_k,
                intermediate_size=self._I_pad,
                # Fixed bound, not the current batch: this selects the kernel's
                # tile configuration, so varying it per call re-selects every step and
                # measured 4x slower than the Triton path.
                tune_max_num_tokens=self._FI_TUNE_MAX_TOKENS,
            )
            if self._H_pad != self.hidden_size:
                # Slicing the padded hidden axis leaves a stride-H_pad view; the
                # moe_forward custom op's meta function declares a contiguous result,
                # so materialise it rather than returning the view.
                output = output[:, :self.hidden_size].contiguous()
            if self.tp_size > 1:
                output = self.allreduce(output)
            return output.view(orig_shape)

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
        if self._use_custom_op:
            return torch.ops.fastkernels.moe_forward(hidden_states, self._layer_name)
        return self.forward_impl(hidden_states)
