"""Semantic PyTorch reference for gla_attention.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.

The production L2 module dispatches to FLA Triton kernels for decode/prefill.
This reference always uses the naive PyTorch recurrence path.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from kb_nano.tasks.baseline.L1.linear import Linear
from kb_nano.tasks.baseline.L1.log_sigmoid import LogSigmoid
from kb_nano.tasks.baseline.L1.silu import SiLU
from kb_nano.tasks.reference.L1.gla_recurrence import NaiveRecurrentGLA
from kb_nano.tasks.reference.L1.rms_norm import RMSNorm
from kb_nano.tasks.reference.L1.rotary_emb import RotaryEmbedding


class GatedLinearAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        decay_mode: Literal["learned_low_rank", "fixed_per_head"] = "learned_low_rank",
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: int = 16,
        use_rotary: bool = False,
        rotary_base: float = 10000.0,
        rotary_max_position: int = 8192,
        norm_eps: float = 1e-6,
        use_fast_kernels: bool = True,
    ):
        super().__init__()
        del use_fast_kernels
        self.num_heads = num_heads
        self.decay_mode = decay_mode
        self.use_rotary = use_rotary
        self.gate_logit_normalizer = gate_logit_normalizer
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.q_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = Linear(self.value_dim, hidden_size, bias=False)
        if decay_mode == "learned_low_rank":
            self.gk_proj = nn.Sequential(
                Linear(hidden_size, gate_low_rank_dim, bias=False),
                Linear(gate_low_rank_dim, self.key_dim, bias=True),
            )
            self.log_sigmoid = LogSigmoid()
        else:
            h_idx = torch.arange(num_heads, dtype=torch.float32)
            gamma = 1.0 - torch.pow(torch.tensor(2.0, dtype=torch.float32), -5.0 - h_idx)
            self.register_buffer("log_gamma", torch.log(gamma), persistent=False)
        if use_rotary:
            self.rotary_emb = RotaryEmbedding(
                head_dim=self.head_k_dim,
                max_position_embeddings=rotary_max_position,
                rope_theta=rotary_base,
            )
        self.naive_recurrence = NaiveRecurrentGLA()
        self.g_norm_swish_gate = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.gate_act = SiLU()

    def _compute_gk(self, hidden_states: torch.Tensor, b: int, t: int) -> torch.Tensor:
        if self.decay_mode == "learned_low_rank":
            gk = self.gk_proj(hidden_states)
            gk = self.log_sigmoid(gk) / self.gate_logit_normalizer
            return gk.view(b, t, self.num_heads, self.head_k_dim).transpose(1, 2)
        return self.log_gamma.to(hidden_states.dtype).view(
            1, self.num_heads, 1, 1,
        ).expand(b, self.num_heads, t, self.head_k_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        del attention_mask, kwargs
        b, t, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)
        if self.use_rotary:
            offsets = getattr(past_key_values, "seq_offsets", None) if past_key_values is not None else None
            local = torch.arange(t, device=q.device, dtype=torch.int64)
            if offsets is None:
                positions = local.repeat(b)
            elif isinstance(offsets, int):
                positions = (local + offsets).repeat(b)
            else:
                positions = (offsets.to(device=q.device, dtype=torch.int64).unsqueeze(1) + local.unsqueeze(0)).reshape(-1)
            q_flat = q.reshape(b * t, self.num_heads * self.head_k_dim).contiguous()
            k_flat = k.reshape(b * t, self.num_heads * self.head_k_dim).contiguous()
            q_flat, k_flat = self.rotary_emb(positions.contiguous(), q_flat, k_flat)
            q = q_flat.view(b, t, self.num_heads, self.head_k_dim)
            k = k_flat.view(b, t, self.num_heads, self.head_k_dim)
        else:
            q = q.view(b, t, self.num_heads, self.head_k_dim)
            k = k.view(b, t, self.num_heads, self.head_k_dim)
        v = v.view(b, t, self.num_heads, self.head_v_dim)
        initial_state = None
        if past_key_values is not None and getattr(past_key_values, "states", None):
            initial_state = past_key_values.states.get(id(self))
        out, final_state = self.naive_recurrence(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            self._compute_gk(hidden_states, b, t),
            initial_state=initial_state,
            output_final_state=use_cache,
        )
        out = out.transpose(1, 2)
        if use_cache and past_key_values is not None:
            if not hasattr(past_key_values, "states"):
                past_key_values.states = {}
            past_key_values.states[id(self)] = final_state
        out = self.g_norm_swish_gate(out.reshape(-1, self.head_v_dim))
        out = out.view(b, t, self.value_dim)
        out = out * self.gate_act(g)
        return self.o_proj(out), None, past_key_values
