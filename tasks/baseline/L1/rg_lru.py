"""RG-LRU (Real-Gated Linear Recurrent Unit) L1 op.

Faithful re-implementation of HF's ``RecurrentGemmaRglru`` from
``transformers/models/recurrent_gemma/modeling_recurrent_gemma.py``,
which is the recurrence used by Google's Griffin/Hawk-style
``recurrent_gemma`` model.

This is a pure-PyTorch eager-scan L1 op: no fused kernel yet. Mirrors
the HF impl exactly so a kb-nano port of recurrent_gemma can replace
``nn.Module``-equivalent ``RecurrentGemmaRglru`` with this L1 op
without changing call sites or state_dict keys.

Parameter naming matches HF (``recurrent_param``, ``input_gate_weight``,
``input_gate_bias``, ``recurrent_gate_weight``, ``recurrent_gate_bias``),
so a state_dict from a ``recurrent_gemma`` reference checkpoint loads
with no remapping.

Future kernel-optimization work would replace ``_rnn_scan`` with a fused
Triton/CUDA sequential-scan kernel (similar pattern to fla's
``fused_recurrent_*`` ops in kb-nano L1). The op interface stays the same.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Matches HF's _MAX_SQRT_GRADIENT (modeling_recurrent_gemma.py uses this constant)
_MAX_SQRT_GRADIENT = 1000.0


class _SqrtBoundDerivative(torch.autograd.Function):
    """``sqrt`` with gradient clipped at ``_MAX_SQRT_GRADIENT``.

    Bit-identical forward to ``torch.sqrt``; backward clips ``1 / sqrt(4 * max(x, 1/MG^2))``
    to prevent NaNs during bf16 training (matches HF's ``SqrtBoundDerivative``).
    Inference-only callers see no behavioral difference (no backward run).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT ** 2))
        return grad_output / torch.sqrt(clipped_x_times_4)


class RGLRU(nn.Module):
    """Real-Gated Linear Recurrent Unit (Griffin / Hawk recurrence).

    Args:
        num_attention_heads: number of recurrent heads (config.num_attention_heads).
        lru_width: total recurrent width (config.lru_width). Must be divisible by num_attention_heads.

    Forward signature matches HF's RecurrentGemmaRglru.forward:
        forward(activations, position_ids) -> hidden_states

    State (held internally, mutable across calls — matches HF):
        self.recurrent_states: optional carry-over recurrent state from a prior call.
            Initialized to None; set to a [batch_size, lru_width] tensor after the
            first forward; used as initial state on subsequent calls (autoregressive
            decoding). Set to None to reset.
    """

    def __init__(self, num_attention_heads: int, lru_width: int):
        super().__init__()
        if lru_width % num_attention_heads != 0:
            raise ValueError(
                f"lru_width ({lru_width}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.lru_width = lru_width
        self.block_width = lru_width // num_attention_heads

        self.recurrent_param = nn.Parameter(torch.empty([lru_width]))
        self.input_gate_weight = nn.Parameter(
            torch.empty([num_attention_heads, self.block_width, self.block_width])
        )
        self.input_gate_bias = nn.Parameter(torch.empty([num_attention_heads, self.block_width]))
        self.recurrent_gate_weight = nn.Parameter(
            torch.empty([num_attention_heads, self.block_width, self.block_width])
        )
        self.recurrent_gate_bias = nn.Parameter(torch.empty([num_attention_heads, self.block_width]))
        self.recurrent_states: torch.Tensor | None = None

    def forward(
        self,
        activations: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, lru_width = activations.shape
        # reset[b, t, 0] is True when position_ids[b, t] == 0 (document boundary)
        reset = position_ids[:, :, None] == 0

        # Per-head gate computation via baddbmm.
        # reshape_act: [num_heads, batch*seq_len, block_width]
        reshape_act = activations.reshape(batch_size * seq_len, self.num_attention_heads, self.block_width)
        reshape_act = reshape_act.permute(1, 0, 2)

        # Input gate: sigmoid(act @ Wi + bi)
        res = torch.baddbmm(self.input_gate_bias[:, None, :], reshape_act, self.input_gate_weight)
        input_gate = torch.sigmoid(res.transpose(0, 1).reshape(batch_size, seq_len, lru_width))

        # Recurrent gate: sigmoid(act @ Wr + br)
        res = torch.baddbmm(self.recurrent_gate_bias[:, None, :], reshape_act, self.recurrent_gate_weight)
        recurrent_gate = torch.sigmoid(res.transpose(0, 1).reshape(batch_size, seq_len, lru_width))

        # Compute the parameter `A` of the recurrence:
        # log_recurrent_gate = -8.0 * recurrent_gate * softplus(recurrent_param)
        # → A = exp(log_recurrent_gate); a_square = A^2 = exp(2 * log_recurrent_gate)
        log_recurrent_gate = -8.0 * recurrent_gate * F.softplus(self.recurrent_param)
        recurrent_gate = torch.exp(log_recurrent_gate)
        a_square = torch.exp(2 * log_recurrent_gate)

        # Gate the input.
        gated_inputs = activations * input_gate

        # Apply gamma normalization to the input.
        # multiplier = sqrt(1 - a_square); reset positions use 1.0 multiplier.
        multiplier = _SqrtBoundDerivative.apply(1 - a_square)
        multiplier = reset + ~reset * multiplier
        normalized_x = gated_inputs * multiplier.type(activations.dtype)

        # Sequential scan (or single-step in sampling mode).
        hidden_states, new_recurrent_states = self._rnn_scan(
            hidden_states=normalized_x,
            recurrent_gate=recurrent_gate,
            reset=reset,
            recurrent_states=self.recurrent_states,
        )
        self.recurrent_states = new_recurrent_states
        return hidden_states

    @staticmethod
    def _rnn_scan(
        hidden_states: torch.Tensor,
        recurrent_gate: torch.Tensor,
        reset: torch.Tensor,
        recurrent_states: torch.Tensor | None,
        acc_dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the linear-RNN recurrence.

        Mirrors HF's RecurrentGemmaRglru._rnn_scan exactly.

        Args:
            hidden_states: input sequence, [B, T, lru_width].
            recurrent_gate: diagonal of the recurrence matrix A, [B, T, lru_width].
            reset: doc-boundary indicator (zero out recurrent_states), [B, T, 1] bool.
            recurrent_states: initial hidden state, [B, lru_width] or None.
            acc_dtype: accumulator dtype (default fp32 — bf16 carry over time accumulates error).
        Returns:
            (output_sequence, final_recurrent_state)
        """
        # Multiply `a` by ~reset (i.e., zero out the recurrence at reset positions).
        recurrent_gate = recurrent_gate * ~reset

        if hidden_states.shape[1] == 1:
            # Sampling mode: single-step decode.
            if recurrent_states is None:
                # First step with no prior state: pass-through, save first step as state.
                return hidden_states, hidden_states[:, 0].type(acc_dtype)
            else:
                contextualized = recurrent_gate.type(acc_dtype) * recurrent_states[:, None].to(recurrent_gate.device)
                contextualized += hidden_states.type(acc_dtype)
                return contextualized.type(hidden_states.dtype), contextualized[:, -1]

        # Linear mode: full sequential scan over T.
        if recurrent_states is None:
            recurrent_states = torch.zeros(
                hidden_states[:, 0].shape, dtype=acc_dtype, device=hidden_states.device
            )

        contextualized = torch.zeros_like(hidden_states)
        for t in range(hidden_states.shape[1]):
            recurrent_states = recurrent_gate[:, t].type(acc_dtype) * recurrent_states.to(recurrent_gate.device)
            recurrent_states = recurrent_states + hidden_states[:, t].type(acc_dtype)
            contextualized[:, t] = recurrent_states.type(hidden_states.dtype)
        return contextualized, recurrent_states
