"""Jamba's Mamba v1 selective-scan mixer (with per-layer dt/B/C RMSNorms).

Reference: ``transformers.models.jamba.modeling_jamba.JambaMambaMixer``
            and ``vllm.model_executor.layers.mamba.mamba_mixer.MambaMixer``.

Key differences from the plain Mamba v1 mixer
(``L2.mamba_mixer.MambaMixer``):

  * Three additional RMSNorms applied to (dt, B, C) after the x_proj
    split.  ``b_layernorm`` and ``c_layernorm`` use ``hidden_size = 16``
    (= ``mamba_d_state``), which falls outside the multiple-of-32
    range the L1 ``RMSNorm`` CUDA kernel handles correctly, so we use
    the autograd-friendly :class:`L1.rms_norm_native.RMSNormNative`
    everywhere here for safety.
  * Operates on flat varlen token layout (``[intermediate, total_tokens]``
    + ``query_start_loc`` + ``cache_indices``) so we can reuse vLLM's
    SOTA fused Mamba kernels (``causal_conv1d_fn``,
    ``selective_scan_fn``, ``causal_conv1d_update``,
    ``selective_state_update``).  The :class:`infra.jamba_engine.JambaEngine`
    flattens its left-padded ``[batch, seq_len]`` input to this layout
    on every forward.
  * Tensor parallel removed: targeted at single-GPU :class:`JambaEngine`.

L1 ops used (no torch.nn.functional or external libs leaked into L2):

  - ``L1.linear.Linear``                   -- in_proj, x_proj, dt_proj, out_proj
  - ``L1.rms_norm_native.RMSNormNative``   -- dt/B/C layernorms
  - ``L1.silu.SiLU``                       -- (the SSM kernel applies the
                                              gate's silu internally; SiLU is
                                              imported for explicitness only)
  - vLLM ``causal_conv1d_*`` / ``selective_*``  -- single-kernel L1 ops in our
                                                   taxonomy: each is one CUDA
                                                   kernel launch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn,
    selective_state_update,
)

from ..L1.linear import Linear
from ..L1.rms_norm_native import RMSNormNative


class JambaMambaMixer(nn.Module):
    """Mamba v1 mixer with Jamba's per-layer dt/B/C RMSNorms.

    Forward expects a *flat varlen* layout that the engine builds:

      hidden_states_flat: ``[total_tokens, hidden_size]``

      ``query_start_loc``: int32 ``[num_seqs + 1]`` with cumulative
                            token starts (in prefill).  ``None`` in
                            decode.
      ``cache_indices``:   int32 ``[num_seqs]`` -- which slot in
                            ``conv_state`` / ``ssm_state`` each seq owns.
      ``has_initial_state``: bool ``[num_seqs]`` -- True iff this seq
                              already has prior conv/ssm state in its slot.
      ``conv_state``:       fp16/bf16 ``[num_slots, intermediate, K-1]``
      ``ssm_state``:        fp16/bf16 ``[num_slots, intermediate, ssm_state_size]``
      ``mode``:             ``"prefill"`` or ``"decode"`` -- decode uses
                              the single-token kernels and ignores
                              ``query_start_loc``.

    The mixer mutates ``conv_state`` / ``ssm_state`` in place at the
    given slot indices (vLLM kernels do this internally).

    Output: ``[total_tokens, hidden_size]``.
    """

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        time_step_rank: int,
        use_conv_bias: bool,
        use_bias: bool,
        rms_norm_eps: float,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.intermediate_size = intermediate_size
        self.time_step_rank = time_step_rank
        self.use_conv_bias = use_conv_bias
        self.layer_idx = layer_idx

        # in_proj packs [hidden, gate], each of size intermediate_size.
        self.in_proj = Linear(
            hidden_size, intermediate_size * 2, bias=use_bias,
        )

        # Depthwise causal conv1d: parameter layout matches HF's
        # ``nn.Conv1d(intermediate, intermediate, K, groups=intermediate)``
        # which stores ``[intermediate, 1, K]``.  The kernels expect a
        # flat ``[intermediate, K]`` view, so we hold the weight as a
        # bare Parameter and ``.view`` to that shape inside forward.
        self.conv1d_weight = nn.Parameter(
            torch.empty(intermediate_size, 1, conv_kernel_size),
        )
        if use_conv_bias:
            self.conv1d_bias = nn.Parameter(torch.empty(intermediate_size))
        else:
            self.register_parameter("conv1d_bias", None)

        # x_proj: x -> (dt, B, C). dt has rank time_step_rank, B/C have
        # rank ssm_state_size each.
        self.x_proj = Linear(
            intermediate_size, time_step_rank + 2 * ssm_state_size,
            bias=False,
        )
        # dt_proj: time_step_rank -> intermediate.  Bias is added by
        # the selective-scan kernel; we keep it as the param's bias and
        # forward it to the kernel explicitly (using F.linear would
        # leak nn.functional into L2, so we go through Linear.matmul
        # without the bias instead).
        self.dt_proj = Linear(time_step_rank, intermediate_size, bias=True)

        # A_log -> A done at load time via the param's weight_loader.
        self.A = nn.Parameter(
            torch.empty(intermediate_size, ssm_state_size, dtype=torch.float32),
        )
        self.A.weight_loader = self._A_loader

        self.D = nn.Parameter(torch.ones(intermediate_size))

        self.out_proj = Linear(intermediate_size, hidden_size, bias=use_bias)

        # Per-layer norms on (dt, B, C). All dims are <= 32 here, which
        # falls outside the L1 RMSNorm CUDA kernel's safe range, so use
        # the native pure-PyTorch path everywhere.
        self.dt_layernorm = RMSNormNative(time_step_rank, eps=rms_norm_eps)
        self.b_layernorm = RMSNormNative(ssm_state_size, eps=rms_norm_eps)
        self.c_layernorm = RMSNormNative(ssm_state_size, eps=rms_norm_eps)

    @staticmethod
    def _A_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Convert HF's stored ``A_log`` to the kernel's expected ``A``.

        The HF checkpoint stores ``log(A)`` (positive); the SSM kernels
        consume ``A = -exp(A_log)`` (negative).  Matches vLLM's loader.
        """
        param.data.copy_(-torch.exp(loaded_weight.float()))

    def _ssm_transform(self, x: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Compute (dt, B, C) from x via x_proj + dt/B/C norms + dt_proj.

        ``x``: ``[total_tokens, intermediate]``
        Returns:
          dt: ``[intermediate, total_tokens]``
          B:  ``[total_tokens, ssm_state_size]``
          C:  ``[total_tokens, ssm_state_size]``
        """
        ssm_params = self.x_proj(x)
        dt, B, C = torch.split(
            ssm_params,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        dt = self.dt_layernorm(dt)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)
        # dt_proj WITHOUT its bias (the kernel adds it). ``F.linear``
        # would be cleaner but is forbidden in L2; go through the L1
        # Linear op's underlying ``Matmul`` directly.
        dt = self.dt_proj.matmul(dt, self.dt_proj.weight, None)
        # Kernels expect ``dt`` as ``[intermediate, total_tokens]``.
        return dt.transpose(-2, -1), B, C

    def forward(
        self,
        hidden_states: torch.Tensor,        # [total_tokens, hidden_size]
        conv_state: torch.Tensor,           # [num_slots, intermediate, K-1]
        ssm_state: torch.Tensor,            # [num_slots, intermediate, ssm_state_size]
        cache_indices: torch.Tensor,        # int32 [num_seqs]
        query_start_loc: torch.Tensor | None,   # int32 [num_seqs + 1] (prefill only)
        has_initial_state: torch.Tensor | None,  # bool [num_seqs] (prefill only)
        is_decode: bool,
        mamba_pad_mask: torch.Tensor | None = None,  # bool [total_tokens] -- True for valid
    ) -> torch.Tensor:
        """Mamba v1 selective-scan forward over a flat varlen batch.

        Returns ``[total_tokens, hidden_size]``.
        """
        # 1. Gated MLP-style input projection: produces [total, intermediate*2]
        projected = self.in_proj(hidden_states)
        # The Mamba kernels expect [intermediate, total_tokens].
        projected = projected.transpose(-2, -1)
        x_states, gate = projected.chunk(2, dim=-2)

        # Zero out padded positions so they don't pollute the recurrent
        # state.  Matches HF's behaviour during left-padded prefill.
        if mamba_pad_mask is not None:
            mask_row = mamba_pad_mask.to(x_states.dtype).unsqueeze(0)
            x_states = x_states * mask_row

        # ``causal_conv1d_*`` expects the weight as [intermediate, K].
        conv_w = self.conv1d_weight.view(
            self.conv1d_weight.size(0), self.conv1d_weight.size(2),
        )
        conv_b = self.conv1d_bias

        if is_decode:
            # x_states is [intermediate, num_decode_seqs] (T=1 each).
            # ``causal_conv1d_update`` expects ``[num_decode, intermediate]``.
            x_t = x_states.transpose(0, 1).contiguous()
            conv_out_t = causal_conv1d_update(
                x_t,
                conv_state,
                conv_w,
                conv_b,
                "silu",
                conv_state_indices=cache_indices,
            )  # [num_decode, intermediate]
            conv_out = conv_out_t.transpose(0, 1).contiguous()  # [inter, num_d]
        else:
            # Prefill: ``causal_conv1d_fn`` expects [intermediate, total],
            # writes per-seq state to ``conv_states[cache_indices[i]]``.
            conv_out = causal_conv1d_fn(
                x_states,
                conv_w,
                conv_b,
                conv_states=conv_state,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                has_initial_state=has_initial_state,
                activation="silu",
            )

        if mamba_pad_mask is not None:
            mask_row = mamba_pad_mask.to(conv_out.dtype).unsqueeze(0)
            conv_out = conv_out * mask_row

        # 2. SSM transform on the post-conv output: (dt, B, C).  The
        # ``_ssm_transform`` expects [total, intermediate], so we
        # transpose into and out of it.
        dt, B_bts, C_bts = self._ssm_transform(conv_out.transpose(-2, -1))

        time_proj_bias = (
            self.dt_proj.bias.float() if self.dt_proj.bias is not None else None
        )

        if is_decode:
            # Single-step state update.  Shapes:
            #   conv_out:    [intermediate, num_decode] -> transpose to [n_d, intermediate]
            #   dt:          [intermediate, num_decode] -> transpose to [n_d, intermediate]
            #   B_bts/C_bts: [num_decode, ssm_state_size]
            #   gate:        [intermediate, num_decode] -> transpose
            scan_out = torch.empty_like(conv_out.transpose(0, 1))  # [n_d, intermediate]
            selective_state_update(
                ssm_state,
                conv_out.transpose(0, 1).contiguous(),
                dt.transpose(0, 1).contiguous(),
                self.A,
                B_bts,
                C_bts,
                self.D,
                gate.transpose(0, 1).contiguous(),
                time_proj_bias,
                dt_softplus=True,
                state_batch_indices=cache_indices,
                out=scan_out,
            )
            scan_out = scan_out.transpose(0, 1).contiguous()  # [inter, n_d]
        else:
            # Prefill scan over the full flat sequence.
            scan_out = selective_scan_fn(
                conv_out,
                ssm_state,
                dt,
                self.A,
                B_bts.transpose(-2, -1),
                C_bts.transpose(-2, -1),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                cache_indices=cache_indices,
                has_initial_state=has_initial_state,
                query_start_loc=query_start_loc,
            )

        # 3. Final out projection.  scan_out is [intermediate, total].
        return self.out_proj(scan_out.transpose(-2, -1))
