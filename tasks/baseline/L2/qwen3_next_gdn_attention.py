"""Qwen3-Next Gated Delta Net (GDN) linear attention.

Implements the GDN block used in Qwen3-Next linear attention layers:
  x -> in_proj_qkvz -> [q,k,v,z]
  x -> in_proj_ba   -> [b,a]
  mixed_qkv = cat(q,k,v) -> causal_conv1d (SiLU) -> split q,k,v
  g = -exp(A_log) * softplus(a + dt_bias)   (forget gate, per v-head)
  beta = sigmoid(b)                          (learning rate, per v-head)
  q,k expanded to num_v_heads if num_v_heads > num_k_heads
  o = chunk_gated_delta_rule(q, k, v, g, beta)
  o = FusedRMSNormGated(o, z)
  o = out_proj(o)

Key dimensions (80B-A3B defaults):
  num_k_heads=16, num_v_heads=32, head_k_dim=128, head_v_dim=128
  key_dim=2048, value_dim=4096, conv_dim=8192
  conv_kernel_size=4

Weight names match HuggingFace checkpoint:
  linear_attn.in_proj_qkvz.weight   [2*key_dim + 2*value_dim, hidden_size]
  linear_attn.in_proj_ba.weight     [2*num_v_heads, hidden_size]  (merged b+a)
  linear_attn.conv1d.weight         [conv_dim, 1, kernel_size]
  linear_attn.A_log                 [num_v_heads]
  linear_attn.dt_bias               [num_v_heads]
  linear_attn.norm.weight           [head_v_dim]
  linear_attn.out_proj.weight       [hidden_size, value_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules.convolution import causal_conv1d

# Use the EXACT same kernels as vLLM for token-level alignment:
# - Prefill: FlashInfer's chunk_gated_delta_rule (H200 compute capability 90)
# - Decode:  vLLM's bundled FLA fused_recurrent_gated_delta_rule
# - Norm:    vLLM's bundled RMSNormGated (norm_before_gate=True)
import triton as _triton
_triton.set_allocator(
    lambda size, alignment, stream: torch.empty(
        size, device="cuda", dtype=torch.int8
    )
)
from flashinfer.gdn_prefill import (
    chunk_gated_delta_rule as _fi_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule as _vllm_fused_recurrent,
)
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
from vllm.model_executor.layers.fla.ops.layernorm_guard import (
    rmsnorm_fn as _vllm_rmsnorm_fn,
)

from ....infra.tp import _tp_size, _tp_rank
from .parallel_linear import ColumnParallelLinear, RowParallelLinear


class _Conv1dWeight(nn.Module):
    """Wrapper to hold conv1d weight as module.weight for correct naming.

    The checkpoint conv1d weight has channels ordered as [Q(key_dim), K(key_dim), V(value_dim)].
    For TP, each segment must be sharded separately since the mixed_qkv input
    is [Q_local, K_local, V_local] per rank.
    """

    def __init__(self, channels: int, kernel_size: int,
                 segment_sizes: list[int] | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        self._segment_sizes = segment_sizes
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight):
        """Load conv1d weight [channels, 1, kernel] -> [channels_local, kernel]."""
        if loaded_weight.dim() == 3:
            loaded_weight = loaded_weight.squeeze(1)
        tp, rank = _tp_size(), _tp_rank()

        if self._segment_sizes is None or tp == 1:
            shard = param.data.size(0)
            param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))
            return

        # Shard each segment (Q, K, V) separately
        offset_src = 0
        offset_dst = 0
        for seg_size in self._segment_sizes:
            local_seg = seg_size // tp
            src = loaded_weight.narrow(0, offset_src + rank * local_seg, local_seg)
            param.data[offset_dst:offset_dst + local_seg].copy_(src)
            offset_src += seg_size
            offset_dst += local_seg


class Qwen3NextGDNAttention(nn.Module):
    """Gated Delta Net linear attention for Qwen3-Next."""

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        tp = _tp_size()
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.local_k_heads = num_k_heads // tp
        self.local_v_heads = num_v_heads // tp
        self.v_per_k = num_v_heads // num_k_heads
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_kernel_size = conv_kernel_size

        # in_proj_qkvz: projects to [Q, K, V, Z] organized by K head groups
        # Layout per K head group: [Q_k_dim, K_k_dim, V_v_per_k*v_dim, Z_v_per_k*v_dim]
        qkvz_dim = 2 * self.key_dim + 2 * self.value_dim
        self.in_proj_qkvz = ColumnParallelLinear(hidden_size, qkvz_dim)

        # in_proj_ba: projects to [b, a] each of size num_v_heads
        # Match vLLM's MergedColumnParallelLinear(output_sizes=[num_v_heads]*2)
        # which splits at midpoint then TP-shards each half independently.
        self.in_proj_ba = ColumnParallelLinear(hidden_size, 2 * num_v_heads)
        self.in_proj_ba.weight.weight_loader = self._ba_weight_loader

        # Causal conv1d on concatenated [Q, K, V]
        conv_dim = 2 * self.key_dim + self.value_dim
        local_conv_dim = conv_dim // tp
        self.conv1d = _Conv1dWeight(
            local_conv_dim, conv_kernel_size,
            segment_sizes=[self.key_dim, self.key_dim, self.value_dim],
        )

        # Decay parameters (sharded across TP)
        self.A_log = nn.Parameter(torch.empty(self.local_v_heads))
        self.A_log.weight_loader = self._sharded_weight_loader
        self.dt_bias = nn.Parameter(torch.empty(self.local_v_heads))
        self.dt_bias.weight_loader = self._sharded_weight_loader

        # Output norm: RMSNorm(x) * silu(z) with norm_before_gate=True
        # Use a simple weight parameter + vLLM's rmsnorm_fn for exact alignment
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(head_v_dim))
        self.norm.eps = rms_norm_eps

        # Output projection
        self.out_proj = RowParallelLinear(self.value_dim, hidden_size)

    @staticmethod
    def _sharded_weight_loader(param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    def _ba_weight_loader(self, param, loaded_weight):
        """Load in_proj_ba weight matching vLLM's MergedColumnParallelLinear.

        vLLM uses output_sizes=[num_v_heads, num_v_heads] which splits the
        [2*num_v_heads, hidden_size] weight at the midpoint, then TP-shards
        each half independently. This creates non-contiguous V-head assignments
        per rank but matches vLLM's exact behavior for token-level alignment.
        """
        tp, rank = _tp_size(), _tp_rank()
        if tp == 1:
            param.data.copy_(loaded_weight)
            return
        half = loaded_weight.size(0) // 2  # num_v_heads
        shard_size = half // tp
        # First half (b shard): take rank's portion
        part0 = loaded_weight.narrow(0, rank * shard_size, shard_size)
        # Second half (a shard): take rank's portion
        part1 = loaded_weight.narrow(0, half + rank * shard_size, shard_size)
        param.data.copy_(torch.cat([part0, part1], dim=0))

    def _unpack_qkvz(self, proj_out):
        """Unpack in_proj_qkvz output into q, k, v, z.

        Input: [N, local_k_heads * per_group_dim]
        where per_group_dim = head_k_dim + head_k_dim + v_per_k*head_v_dim + v_per_k*head_v_dim

        Returns: q [N, local_k_heads, head_k_dim],
                 k [N, local_k_heads, head_k_dim],
                 v [N, local_v_heads, head_v_dim],
                 z [N, local_v_heads, head_v_dim]
        """
        N = proj_out.shape[0]
        per_group = (
            self.head_k_dim + self.head_k_dim
            + self.v_per_k * self.head_v_dim
            + self.v_per_k * self.head_v_dim
        )
        # Reshape to [N, local_k_heads, per_group]
        x = proj_out.view(N, self.local_k_heads, per_group)

        q = x[:, :, :self.head_k_dim]  # [N, Hk, Dk]
        off = self.head_k_dim
        k = x[:, :, off:off + self.head_k_dim]  # [N, Hk, Dk]
        off += self.head_k_dim
        v_size = self.v_per_k * self.head_v_dim
        v = x[:, :, off:off + v_size]  # [N, Hk, v_per_k * Dv]
        off += v_size
        z = x[:, :, off:off + v_size]  # [N, Hk, v_per_k * Dv]

        # Reshape V, Z to [N, local_v_heads, head_v_dim]
        v = v.reshape(N, self.local_v_heads, self.head_v_dim)
        z = z.reshape(N, self.local_v_heads, self.head_v_dim)

        return q, k, v, z

    def _unpack_ba(self, proj_out):
        """Unpack in_proj_ba output into b, a.

        Input: [N, 2 * local_v_heads] (organized by K head groups)
        Returns: b [N, local_v_heads], a [N, local_v_heads]
        """
        N = proj_out.shape[0]
        # The BA projection is organized by K head groups too
        # Each K head group has v_per_k b values and v_per_k a values
        x = proj_out.view(N, self.local_k_heads, 2 * self.v_per_k)
        b = x[:, :, :self.v_per_k].reshape(N, self.local_v_heads)
        a = x[:, :, self.v_per_k:].reshape(N, self.local_v_heads)
        return b, a

    def forward(self, hidden_states, layer_state=None):
        """
        Args:
            hidden_states: [B, T, hidden_size]
            layer_state: dict with 'conv' and 'recurrent' state, or None
        Returns:
            output: [B, T, hidden_size]
        """
        B, T, _ = hidden_states.shape
        x_flat = hidden_states.reshape(-1, self.hidden_size)
        N = x_flat.shape[0]

        # 1. Input projections
        qkvz_out = self.in_proj_qkvz(x_flat)  # [N, qkvz_local]
        ba_out = self.in_proj_ba(x_flat)  # [N, 2*local_v_heads]

        q, k, v, z = self._unpack_qkvz(qkvz_out)
        b, a = self._unpack_ba(ba_out)

        # 2. Causal conv1d on concatenated [q, k, v]
        # Flatten q, k heads: q [N, Hk*Dk], k [N, Hk*Dk], v [N, Hv*Dv]
        q_flat = q.reshape(N, self.local_k_heads * self.head_k_dim)
        k_flat = k.reshape(N, self.local_k_heads * self.head_k_dim)
        v_flat = v.reshape(N, self.local_v_heads * self.head_v_dim)
        mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1)  # [N, conv_dim_local]

        # Reshape to [B, T, D] for causal_conv1d
        mixed_qkv = mixed_qkv.view(B, T, -1)

        conv_state = layer_state.get("conv") if layer_state is not None else None
        output_final_state = (layer_state is not None)
        mixed_qkv, conv_state_new = causal_conv1d(
            mixed_qkv, self.conv1d.weight, activation='silu',
            initial_state=conv_state, output_final_state=output_final_state,
            backend='cuda',
        )
        if layer_state is not None:
            # vLLM stores conv state in bf16 (mamba_cache_dtype = model dtype)
            layer_state["conv"] = conv_state_new.to(torch.bfloat16) if conv_state_new is not None else None

        # Split back into q, k, v
        local_k_dim = self.local_k_heads * self.head_k_dim
        local_v_dim = self.local_v_heads * self.head_v_dim
        q_conv, k_conv, v_conv = mixed_qkv.split(
            [local_k_dim, local_k_dim, local_v_dim], dim=-1
        )

        # Reshape to [B, T, H, D]
        q_4d = q_conv.view(B, T, self.local_k_heads, self.head_k_dim)
        k_4d = k_conv.view(B, T, self.local_k_heads, self.head_k_dim)
        v_4d = v_conv.view(B, T, self.local_v_heads, self.head_v_dim)

        # 3. GVA: q/k have local_k_heads, v has local_v_heads.
        # FLA's chunk_gated_delta_rule handles GVA internally.

        # 4. Compute gating
        # g = -exp(A_log) * softplus(a + dt_bias), per v-head
        a_3d = a.view(B, T, self.local_v_heads)
        g = -self.A_log.float().exp() * F.softplus(
            a_3d.float() + self.dt_bias.float()
        )  # [B, T, Hv_local]

        # beta = sigmoid(b), per v-head
        # Cast to input dtype to match vLLM's fused_gdn_gating output precision
        b_3d = b.view(B, T, self.local_v_heads)
        beta = b_3d.float().sigmoid().to(b_3d.dtype)  # [B, T, Hv_local]

        # 5. Recurrence
        recurrent_state = layer_state.get("recurrent") if layer_state is not None else None
        want_final_state = (layer_state is not None)
        # vLLM uses chunk (FlashInfer) for all prefills, fused_recurrent for decodes
        mode = 'fused_recurrent' if T == 1 else 'chunk'

        if mode == 'chunk':
            # Match vLLM's FlashInfer path exactly:
            #   l2norm q,k externally, exp(g) for linear-space gating,
            #   float32 for g/beta/state, squeeze batch dim for FlashInfer.
            q_norm = l2norm_fwd(q_4d.contiguous())
            k_norm = l2norm_fwd(k_4d.contiguous())

            fi_g = g.squeeze(0).contiguous().float()
            fi_beta = beta.float().squeeze(0).contiguous()

            if recurrent_state is None:
                fi_state = torch.zeros(
                    1, self.local_v_heads, self.head_v_dim, self.head_k_dim,
                    device=q_4d.device, dtype=torch.float32,
                )
            else:
                fi_state = recurrent_state.to(torch.float32)

            cu_seqlens = torch.tensor(
                [0, T], dtype=torch.int32, device=q_4d.device,
            )
            fi_result = _fi_chunk_gated_delta_rule(
                q=q_norm.squeeze(0).contiguous(),
                k=k_norm.squeeze(0).contiguous(),
                v=v_4d.squeeze(0).contiguous(),
                g=torch.exp(fi_g),
                beta=fi_beta,
                initial_state=fi_state,
                output_final_state=want_final_state,
                cu_seqlens=cu_seqlens,
            )
            if want_final_state:
                o, recurrent_state_new = fi_result
            else:
                o, recurrent_state_new = fi_result, None
            o = o.unsqueeze(0)  # restore batch dim [1, T, H, D]
        else:
            # Decode path: use vLLM's bundled fused_recurrent for exact alignment.
            # vLLM passes bf16 ssm_state directly with inplace_final_state=True.
            # We match by keeping state in bf16 throughout.
            if recurrent_state is None:
                fr_state = torch.zeros(
                    1, self.local_v_heads, self.head_v_dim, self.head_k_dim,
                    device=q_4d.device, dtype=torch.bfloat16,
                )
            else:
                fr_state = recurrent_state  # already bf16

            cu_seqlens = torch.tensor(
                [0, T], dtype=torch.int64, device=q_4d.device,
            )
            o, recurrent_state_new = _vllm_fused_recurrent(
                q=q_4d, k=k_4d, v=v_4d,
                g=g, beta=beta,
                initial_state=fr_state,
                inplace_final_state=False,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )

        if layer_state is not None and recurrent_state_new is not None:
            # vLLM stores recurrent state in bf16 (ssm_state cache dtype = model dtype)
            layer_state["recurrent"] = recurrent_state_new.to(torch.bfloat16)

        # 6. Output gating: RMSNorm(o) * silu(z) using vLLM's rmsnorm_fn
        # o: [B, T, Hv_local, Dv], z: [N, Hv_local, Dv]
        z_4d = z.view(B, T, self.local_v_heads, self.head_v_dim)
        o_flat = o.reshape(-1, self.head_v_dim)
        z_flat = z_4d.reshape(-1, self.head_v_dim)
        o = _vllm_rmsnorm_fn(
            o_flat, self.norm.weight, bias=None,
            z=z_flat, eps=self.norm.eps,
            norm_before_gate=True, activation='swish',
        ).view(B, T, self.local_v_heads, self.head_v_dim)

        # 7. Output projection
        o = o.reshape(B * T, self.local_v_heads * self.head_v_dim)
        return self.out_proj(o).view(B, T, self.hidden_size)
