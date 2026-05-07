"""KDA (Kimi Delta Attention) layer (L2).

Implements the KDA block used in Kimi-Linear:
  x -> q/k/v proj -> causal conv1d (SiLU) -> Delta-Net recurrence
    -> RMSNorm + sigmoid gate -> o proj

Operates on a flat varlen batch ``[num_actual_tokens, hidden_size]`` with
per-request metadata (``KimiLinearMetadata``) supplied by the engine.
Decode requests come first in the batch, then prefill requests, so the
fused-recurrent decode path can slice ``query_start_loc[: num_decodes + 1]``
without any per-sequence Python loops -- mirroring vLLM's
``KimiDeltaAttention._forward`` (``vllm/model_executor/layers/kda.py``).

State (conv windows + Delta-Net recurrent matrix) lives in flat per-layer
tensors owned by ``KimiLinearStateManager``; the layer reads/writes its
own slice via the per-request ``state_indices``.

Composes only L1 ops (``CausalConv1d``, ``KDARecurrence``, ``KDAGate``,
``RMSNorm``) and the canonical TP linears in ``parallel_linear``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.kimi_linear_metadata import get_metadata
from ....infra.tp import _tp_size, _tp_rank
from ..L1.causal_conv1d import CausalConv1d
from ..L1.kda_recurrence import KDAGate, KDARecurrence
from ..L1.rms_norm import RMSNorm
from .parallel_linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from fla.ops.kda import chunk_kda as _fla_chunk_kda
from fla.ops.kda import fused_recurrent_kda as _fla_fused_recurrent_kda


class KDAAttention(nn.Module):
    """Kimi Delta Attention (linear attention with Delta-Net recurrence)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        layer_idx: int,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        tp = _tp_size()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.local_num_heads = num_heads // tp
        projection_size = num_heads * head_dim
        local_projection = self.local_num_heads * head_dim
        self.local_projection = local_projection
        self.conv_kernel_size = conv_kernel_size
        self.rms_norm_eps = rms_norm_eps

        self.q_proj = ColumnParallelLinear(hidden_size, projection_size)
        self.k_proj = ColumnParallelLinear(hidden_size, projection_size)
        self.v_proj = ColumnParallelLinear(hidden_size, projection_size)
        self.o_proj = RowParallelLinear(projection_size, hidden_size)

        self.q_conv1d = ColumnParallelLinear(conv_kernel_size, projection_size)
        self.q_conv1d.weight.weight_loader = self._conv_weight_loader
        self.k_conv1d = ColumnParallelLinear(conv_kernel_size, projection_size)
        self.k_conv1d.weight.weight_loader = self._conv_weight_loader
        self.v_conv1d = ColumnParallelLinear(conv_kernel_size, projection_size)
        self.v_conv1d.weight.weight_loader = self._conv_weight_loader

        self.f_a_proj = ReplicatedLinear(hidden_size, head_dim, bias=False)
        self.f_b_proj = ColumnParallelLinear(head_dim, projection_size)

        self.dt_bias = nn.Parameter(
            torch.empty(local_projection, dtype=torch.float32)
        )
        self.dt_bias.weight_loader = self._sharded_dim0
        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        self.A_log.weight_loader = self._sharded_dim2

        self.b_proj = ColumnParallelLinear(hidden_size, num_heads)

        self.g_a_proj = ReplicatedLinear(hidden_size, head_dim, bias=False)
        self.g_b_proj = ColumnParallelLinear(head_dim, projection_size)

        self.o_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        self.q_conv = CausalConv1d()
        self.k_conv = CausalConv1d()
        self.v_conv = CausalConv1d()
        self.kda_gate = KDAGate()
        self.kda_recurrence = KDARecurrence()

    @staticmethod
    def _conv_weight_loader(param, loaded_weight):
        if loaded_weight.dim() == 3:
            loaded_weight = loaded_weight.squeeze(1)
        rank = _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    @staticmethod
    def _sharded_dim0(param, loaded_weight):
        rank = _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    @staticmethod
    def _sharded_dim2(param, loaded_weight):
        rank = _tp_rank()
        shard = param.data.size(2)
        param.data.copy_(loaded_weight.narrow(2, rank * shard, shard))

    def _output_norm_gate(self, attn_out: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """RMSNorm(attn_out) * sigmoid(g2). Both are [N, H_local, D]."""
        flat = attn_out.reshape(-1, self.head_dim)
        normed = self.o_norm(flat).view_as(attn_out)
        return normed * torch.sigmoid(g2)

    def _apply_conv(
        self,
        conv_module: CausalConv1d,
        weight: torch.Tensor,
        x: torch.Tensor,
        state_tensor: torch.Tensor,
        state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Run varlen causal_conv1d on a flat ``[N, D]`` token stream.

        Reads the per-request initial state from ``state_tensor[state_indices]``
        and scatters the final state back. Returns ``[N, D]``.
        """
        # fla expects [B, T, D]; we use B=1 with cu_seqlens.
        x_b1 = x.unsqueeze(0)
        init_state = state_tensor.index_select(0, state_indices)
        out, final_state = conv_module(
            x_b1, weight,
            initial_state=init_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        # Scatter updated state back. fla returns [N, D, K]; storage is the same.
        state_tensor.index_copy_(0, state_indices, final_state.to(state_tensor.dtype))
        return out.squeeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_manager=None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [num_actual_tokens, hidden_size] flat varlen batch
            state_manager: ``KimiLinearStateManager`` with this layer's flat
                conv/recurrent state tensors (or ``None`` during warmup).
        Returns:
            output: [num_actual_tokens, hidden_size]
        """
        md = get_metadata()
        N = hidden_states.shape[0]

        if md is None or state_manager is None:
            # Warmup / profile run: zero output, no state mutation.
            return torch.zeros_like(hidden_states)

        layer_idx = self.layer_idx
        # PyTorch's index_select / index_copy_ require int64 indices; the
        # batch metadata stores them as int32 (matching vLLM's GDN layout).
        state_indices = md.state_indices.long()  # [B]

        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)

        beta = self.b_proj(hidden_states).float().sigmoid()

        g1_raw = self.f_b_proj(self.f_a_proj(hidden_states))
        g2_raw = self.g_b_proj(self.g_a_proj(hidden_states))

        cu_q = md.query_start_loc

        q = self._apply_conv(
            self.q_conv, self.q_conv1d.weight, q_proj,
            state_manager.conv_q[layer_idx], state_indices, cu_q,
        )
        k = self._apply_conv(
            self.k_conv, self.k_conv1d.weight, k_proj,
            state_manager.conv_k[layer_idx], state_indices, cu_q,
        )
        v = self._apply_conv(
            self.v_conv, self.v_conv1d.weight, v_proj,
            state_manager.conv_v[layer_idx], state_indices, cu_q,
        )

        H = self.local_num_heads
        D = self.head_dim
        q = q.view(1, N, H, D)
        k = k.view(1, N, H, D)
        v = v.view(1, N, H, D)

        g1 = g1_raw.view(N, H, D)
        g_log = self.kda_gate(g1, self.A_log, self.dt_bias).view(1, N, H, D)
        beta = beta.view(1, N, H)

        recurrent_full = state_manager.recurrent[layer_idx]  # [num_slots, H, D, D]

        attn_out = torch.empty(N, H, D, device=hidden_states.device, dtype=v.dtype)

        # Decode-only block (first num_decodes requests, one token each):
        # use the fused recurrent kernel for low latency.
        nd = md.num_decodes
        if nd > 0:
            n_dec_tok = md.num_decode_tokens
            cu_dec = cu_q[: nd + 1]
            dec_state_idx = state_indices[:nd]
            init_state_dec = recurrent_full.index_select(0, dec_state_idx).contiguous()
            o_dec, final_state_dec = _fla_fused_recurrent_kda(
                q[:, :n_dec_tok].contiguous(),
                k[:, :n_dec_tok].contiguous(),
                v[:, :n_dec_tok].contiguous(),
                g_log[:, :n_dec_tok].contiguous(),
                beta[:, :n_dec_tok].contiguous(),
                initial_state=init_state_dec,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_dec,
            )
            recurrent_full.index_copy_(0, dec_state_idx, final_state_dec.to(recurrent_full.dtype))
            attn_out[:n_dec_tok] = o_dec.squeeze(0)

        # Prefill block (last num_prefills requests): chunked Delta-Net
        # for high throughput. ``has_initial_state`` is per-request batch
        # ordered (decodes + prefills) so we slice the prefill suffix.
        np_ = md.num_prefills
        if np_ > 0:
            n_pf_tok = md.num_prefill_tokens
            cu_pf = cu_q[md.num_decodes:] - cu_q[md.num_decodes]
            pf_state_idx = state_indices[md.num_decodes:]
            has_initial = md.has_initial_state[md.num_decodes:]
            init_state_pf = recurrent_full.index_select(0, pf_state_idx).contiguous()
            # Sequences with no prior state must start from zeros so the
            # kernel can't accidentally use stale data left in the slot.
            zero_mask = (~has_initial).nonzero(as_tuple=True)[0]
            if zero_mask.numel() > 0:
                init_state_pf[zero_mask] = 0
            o_pf, final_state_pf = _fla_chunk_kda(
                q[:, md.num_decode_tokens:].contiguous(),
                k[:, md.num_decode_tokens:].contiguous(),
                v[:, md.num_decode_tokens:].contiguous(),
                g_log[:, md.num_decode_tokens:].contiguous(),
                beta[:, md.num_decode_tokens:].contiguous(),
                initial_state=init_state_pf,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_pf,
            )
            recurrent_full.index_copy_(0, pf_state_idx, final_state_pf.to(recurrent_full.dtype))
            attn_out[md.num_decode_tokens:] = o_pf.squeeze(0)

        g2 = g2_raw.view(N, H, D)
        attn_out = self._output_norm_gate(attn_out, g2)

        return self.o_proj(attn_out.reshape(N, H * D))
