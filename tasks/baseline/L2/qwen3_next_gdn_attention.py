"""Qwen3-Next Gated Delta Net (GDN) linear attention (L2).

Implements the GDN block used in Qwen3-Next linear attention layers:
  x -> in_proj_qkvz -> [q,k,v,z]
  x -> in_proj_ba   -> [b,a]
  mixed_qkv = cat(q,k,v) -> CausalConv1d (SiLU) -> split q,k,v
  g = -exp(A_log) * softplus(a + dt_bias)   (forget gate, per v-head)
  beta = sigmoid(b)                          (learning rate, per v-head)
  q,k expanded to num_v_heads if num_v_heads > num_k_heads
  o = GDNChunkPrefill(q, k, v, g, beta)         [prefill]
   or GDNFusedRecurrent(q, k, v, g, beta)        [decode]
  o = RMSNormGated(o, z)                     (norm_before_gate=True, swish)
  o = out_proj(o)

Composes only L1 ops (``CausalConv1d``, ``GDNChunkPrefill``,
``GDNFusedRecurrent``, ``L2NormFwd``, ``Softplus``, ``RMSNormGated``)
plus the canonical TP linears in ``parallel_linear``. No direct imports
of ``fla``, ``flashinfer``, ``vllm``, or ``torch.nn.functional`` here.

Key dimensions (80B-A3B defaults):
  num_k_heads=16, num_v_heads=32, head_k_dim=128, head_v_dim=128
  key_dim=2048, value_dim=4096, conv_dim=8192
  conv_kernel_size=4

Weight names match HuggingFace checkpoint:
  linear_attn.in_proj_qkvz.weight   [2*key_dim + 2*value_dim, hidden_size]
  linear_attn.in_proj_ba.weight     [2*num_v_heads, hidden_size]
  linear_attn.conv1d.weight         [conv_dim, 1, kernel_size]
  linear_attn.A_log                 [num_v_heads]
  linear_attn.dt_bias               [num_v_heads]
  linear_attn.norm.weight           [head_v_dim]
  linear_attn.out_proj.weight       [hidden_size, value_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size, _tp_rank
from ..L1.causal_conv1d import CausalConv1d
from ..L1.gdn_recurrence import GDNChunkPrefill, GDNFusedRecurrent
from ..L1.l2norm_kernel import L2NormFwd
from ..L1.rms_norm_gated import RMSNormGated
from ..L1.softplus import Softplus
from .parallel_linear import ColumnParallelLinear, RowParallelLinear


class _Conv1dWeight(nn.Module):
    """Parameter container for the GDN causal-conv1d kernel.

    The checkpoint stores conv1d weight as ``[channels, 1, kernel]``
    (``nn.Conv1d`` layout) where channels are organized as
    ``[Q(key_dim), K(key_dim), V(value_dim)]``. For TP, each segment
    must be sharded independently because the runtime input layout per
    rank is ``[Q_local, K_local, V_local]``. ``ColumnParallelLinear``
    can't express that piecewise sharding, so we hold the parameter
    here with a custom loader.
    """

    def __init__(self, channels: int, kernel_size: int,
                 segment_sizes: list[int] | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        self._segment_sizes = segment_sizes
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight):
        if loaded_weight.dim() == 3:
            loaded_weight = loaded_weight.squeeze(1)
        tp, rank = _tp_size(), _tp_rank()

        if self._segment_sizes is None or tp == 1:
            shard = param.data.size(0)
            param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))
            return

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

        # in_proj_ba: projects to [b, a] each of size num_v_heads.
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
        # vLLM's rmsnorm_fn (wrapped here as L1.RMSNormGated) gives bitwise
        # alignment with vLLM's runtime.
        self.norm = RMSNormGated(
            head_v_dim, eps=rms_norm_eps,
            norm_before_gate=True, activation="swish",
        )

        # Output projection
        self.out_proj = RowParallelLinear(self.value_dim, hidden_size)

        # L1 ops
        self.conv = CausalConv1d()
        self.softplus = Softplus()
        self.l2norm = L2NormFwd()
        self.gdn_chunk = GDNChunkPrefill()
        self.gdn_fused_recurrent = GDNFusedRecurrent()

    @staticmethod
    def _sharded_weight_loader(param, loaded_weight):
        rank = _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    def _ba_weight_loader(self, param, loaded_weight):
        """Load in_proj_ba weight matching vLLM's MergedColumnParallelLinear.

        vLLM uses ``output_sizes=[num_v_heads, num_v_heads]`` which splits the
        ``[2*num_v_heads, hidden_size]`` weight at the midpoint, then TP-shards
        each half independently. This creates non-contiguous V-head assignments
        per rank but matches vLLM's exact behavior for token-level alignment.
        """
        tp, rank = _tp_size(), _tp_rank()
        if tp == 1:
            param.data.copy_(loaded_weight)
            return
        half = loaded_weight.size(0) // 2  # num_v_heads
        shard_size = half // tp
        part0 = loaded_weight.narrow(0, rank * shard_size, shard_size)
        part1 = loaded_weight.narrow(0, half + rank * shard_size, shard_size)
        param.data.copy_(torch.cat([part0, part1], dim=0))

    def _unpack_qkvz(self, proj_out):
        """Unpack in_proj_qkvz output into q, k, v, z."""
        N = proj_out.shape[0]
        per_group = (
            self.head_k_dim + self.head_k_dim
            + self.v_per_k * self.head_v_dim
            + self.v_per_k * self.head_v_dim
        )
        x = proj_out.view(N, self.local_k_heads, per_group)

        q = x[:, :, :self.head_k_dim]
        off = self.head_k_dim
        k = x[:, :, off:off + self.head_k_dim]
        off += self.head_k_dim
        v_size = self.v_per_k * self.head_v_dim
        v = x[:, :, off:off + v_size]
        off += v_size
        z = x[:, :, off:off + v_size]

        v = v.reshape(N, self.local_v_heads, self.head_v_dim)
        z = z.reshape(N, self.local_v_heads, self.head_v_dim)
        return q, k, v, z

    def _unpack_ba(self, proj_out):
        """Unpack in_proj_ba output into b, a (organized by K head groups)."""
        N = proj_out.shape[0]
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
        qkvz_out = self.in_proj_qkvz(x_flat)
        ba_out = self.in_proj_ba(x_flat)

        q, k, v, z = self._unpack_qkvz(qkvz_out)
        b, a = self._unpack_ba(ba_out)

        # 2. Causal conv1d on concatenated [q, k, v]
        q_flat = q.reshape(N, self.local_k_heads * self.head_k_dim)
        k_flat = k.reshape(N, self.local_k_heads * self.head_k_dim)
        v_flat = v.reshape(N, self.local_v_heads * self.head_v_dim)
        mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1)
        mixed_qkv = mixed_qkv.view(B, T, -1)

        conv_state = layer_state.get("conv") if layer_state is not None else None
        output_final_state = (layer_state is not None)
        mixed_qkv, conv_state_new = self.conv(
            mixed_qkv, self.conv1d.weight,
            initial_state=conv_state,
            output_final_state=output_final_state,
            backend="cuda",
        )
        if layer_state is not None:
            # vLLM stores conv state in bf16 (mamba_cache_dtype = model dtype)
            layer_state["conv"] = (
                conv_state_new.to(torch.bfloat16) if conv_state_new is not None else None
            )

        local_k_dim = self.local_k_heads * self.head_k_dim
        local_v_dim = self.local_v_heads * self.head_v_dim
        q_conv, k_conv, v_conv = mixed_qkv.split(
            [local_k_dim, local_k_dim, local_v_dim], dim=-1
        )

        q_4d = q_conv.view(B, T, self.local_k_heads, self.head_k_dim)
        k_4d = k_conv.view(B, T, self.local_k_heads, self.head_k_dim)
        v_4d = v_conv.view(B, T, self.local_v_heads, self.head_v_dim)

        # 3. Gating: g = -exp(A_log) * softplus(a + dt_bias) per v-head
        a_3d = a.view(B, T, self.local_v_heads)
        g = -self.A_log.float().exp() * self.softplus(
            a_3d.float() + self.dt_bias.float()
        )

        b_3d = b.view(B, T, self.local_v_heads)
        beta = b_3d.float().sigmoid().to(b_3d.dtype)

        # 4. Recurrence — vLLM uses chunk (FlashInfer) for prefill,
        # fused_recurrent for decode.
        recurrent_state = (
            layer_state.get("recurrent") if layer_state is not None else None
        )
        want_final_state = (layer_state is not None)
        is_decode = (T == 1)

        if not is_decode:
            # FlashInfer chunked path: l2norm q,k externally, exp(g) for
            # linear-space gating, fp32 g/beta/state, batch dim squeezed.
            q_norm = self.l2norm(q_4d.contiguous())
            k_norm = self.l2norm(k_4d.contiguous())

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
            fi_result = self.gdn_chunk(
                q_norm.squeeze(0).contiguous(),
                k_norm.squeeze(0).contiguous(),
                v_4d.squeeze(0).contiguous(),
                torch.exp(fi_g),
                fi_beta,
                initial_state=fi_state,
                output_final_state=want_final_state,
                cu_seqlens=cu_seqlens,
            )
            if want_final_state:
                o, recurrent_state_new = fi_result
            else:
                o, recurrent_state_new = fi_result, None
            o = o.unsqueeze(0)
        else:
            # Decode path: vLLM's bundled FLA fused_recurrent. State is bf16.
            if recurrent_state is None:
                fr_state = torch.zeros(
                    1, self.local_v_heads, self.head_v_dim, self.head_k_dim,
                    device=q_4d.device, dtype=torch.bfloat16,
                )
            else:
                fr_state = recurrent_state

            cu_seqlens = torch.tensor(
                [0, T], dtype=torch.int64, device=q_4d.device,
            )
            o, recurrent_state_new = self.gdn_fused_recurrent(
                q_4d, k_4d, v_4d, g, beta,
                initial_state=fr_state,
                cu_seqlens=cu_seqlens,
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )

        if layer_state is not None and recurrent_state_new is not None:
            layer_state["recurrent"] = recurrent_state_new.to(torch.bfloat16)

        # 5. Output gating: RMSNorm(o) * silu(z) (L1 op)
        z_4d = z.view(B, T, self.local_v_heads, self.head_v_dim)
        o_flat = o.reshape(-1, self.head_v_dim)
        z_flat = z_4d.reshape(-1, self.head_v_dim)
        o = self.norm(o_flat, z_flat).view(
            B, T, self.local_v_heads, self.head_v_dim
        )

        # 6. Output projection
        o = o.reshape(B * T, self.local_v_heads * self.head_v_dim)
        return self.out_proj(o).view(B, T, self.hidden_size)
