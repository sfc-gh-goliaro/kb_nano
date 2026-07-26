"""Vision transformer block for Qwen VL models (L3 composite).

Naive pure-torch reference for ``tasks/baseline/L3/vision_block.py``:
pre-norm residual block wiring VisionAttention (varlen, non-causal,
rotary embedding) and VisionMLP (QuickGELU by default).

Unified across Qwen2-VL and Qwen3-VL:
  - act_fn: Qwen2 uses QuickGELU (default), Qwen3 uses SiLU.
  - norm_eps: configurable LayerNorm epsilon.
"""


from __future__ import annotations

from typing import Callable

# Inlined from tasks/reference/L1/quickgelu.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


# Inlined from tasks/reference/L1/layer_norm.py


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        create_scale: bool = True,
        create_offset: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)
        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(
            x.float(), self.normalized_shape, weight, bias, self.eps,
        ).to(orig_dtype)


# Single-rank stand-ins for the TP linear layers (adapted from
# tasks/reference/L2/parallel_linear.py with tp_size == 1; identical
# parameter names and shapes, so the state_dict tree matches the baseline).


class ColumnParallelLinear(nn.Module):
    """Splits output dim across TP ranks (single-rank here)."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None):
        super().__init__()
        self.output_size_per_partition = output_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Splits input dim across TP ranks (single-rank here, no all-reduce)."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None, reduce_results: bool = True):
        super().__init__()
        self.input_size_per_partition = input_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class QKVParallelLinear(nn.Module):
    """Q, K, V projections merged (single-rank)."""

    def __init__(self, hidden_size: int, head_size: int,
                 total_num_heads: int, total_num_kv_heads: int,
                 bias: bool = False, quant_config: dict | None = None):
        super().__init__()
        self.head_size = head_size
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
        self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


# Inlined from tasks/reference/L1/_attention.py


def repeat_kv(k: torch.Tensor, target_heads: int) -> torch.Tensor:
    if k.shape[-2] == target_heads:
        return k
    if target_heads % k.shape[-2] != 0:
        raise ValueError(
            f"Cannot repeat {k.shape[-2]} KV heads to {target_heads} query heads"
        )
    return k.repeat_interleave(target_heads // k.shape[-2], dim=-2)


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int] | list[int] | None = (-1, -1),
    s_aux: torch.Tensor | None = None,
    softcap: float = 0.0,
) -> torch.Tensor:
    window_size = (-1, -1) if window_size is None else tuple(window_size)
    q_in = q.transpose(-3, -2)
    k_in = repeat_kv(k, q.shape[-2]).transpose(-3, -2)
    v_in = repeat_kv(v, q.shape[-2]).transpose(-3, -2)
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    has_backend_specific_mask = (
        window_size != (-1, -1)
        or s_aux is not None
        or softcap > 0.0
    )
    if q.is_cuda and not has_backend_specific_mask and q_in.shape[-2] == k_in.shape[-2]:
        out = torch.ops.aten._scaled_dot_product_flash_attention(
            q_in, k_in, v_in, 0.0, causal, scale=scale,
        )[0]
        return out.transpose(-3, -2)
    if causal or has_backend_specific_mask:
        q_len = q_in.shape[-2]
        k_len = k_in.shape[-2]
        left, right = window_size
        if causal:
            right = 0
        q_pos = torch.arange(q_len, device=q.device).unsqueeze(1) + (k_len - q_len)
        k_pos = torch.arange(k_len, device=q.device).unsqueeze(0)
        if left < 0:
            mask = k_pos <= q_pos + right
        else:
            mask = (k_pos <= torch.minimum(q_pos + right, torch.full_like(q_pos, k_len))) & (
                k_pos >= q_pos - left
            )
        scores = torch.matmul(q_in.float(), k_in.float().transpose(-2, -1)) * scale
        if softcap > 0.0:
            scores = torch.tanh(scores / softcap) * softcap
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        probs = probs.masked_fill(torch.all(~mask, dim=-1, keepdim=True), 0.0)
        out = torch.matmul(probs.to(v_in.dtype), v_in)
        return out.transpose(-3, -2)
    out = F.scaled_dot_product_attention(
        q_in, k_in, v_in, is_causal=False, scale=scale,
    )
    return out.transpose(-3, -2)


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int] | list[int] | None = (-1, -1),
    s_aux: torch.Tensor | None = None,
    softcap: float = 0.0,
) -> torch.Tensor:
    window_size = (-1, -1) if window_size is None else tuple(window_size)
    outputs = []
    batch = cu_seqlens_q.numel() - 1
    for i in range(batch):
        q_start = int(cu_seqlens_q[i].item())
        q_end = int(cu_seqlens_q[i + 1].item())
        k_start = int(cu_seqlens_k[i].item())
        k_end = int(cu_seqlens_k[i + 1].item())
        out = dense_attention(
            q[q_start:q_end].unsqueeze(0),
            k[k_start:k_end].unsqueeze(0),
            v[k_start:k_end].unsqueeze(0),
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            s_aux=s_aux,
            softcap=softcap,
        ).squeeze(0)
        outputs.append(out)
    if not outputs:
        return q.new_empty(q.shape)
    return torch.cat(outputs, dim=0)


# Inlined from tasks/reference/L1/flash_attn_prefill.py


class FlashAttnPrefill(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        del max_seqlen_q, max_seqlen_k
        window_size = kwargs.get("window_size", (-1, -1))
        window_size = (-1, -1) if window_size is None else tuple(window_size)
        return varlen_attention(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            softmax_scale=kwargs.get("softmax_scale", self.sm_scale),
            causal=kwargs.get("causal", True),
            window_size=window_size,
            s_aux=kwargs.get("s_aux", None),
            softcap=kwargs.get("softcap", 0.0),
        )


# Inlined from tasks/reference/L2/vision_attention.py


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary embedding matching flash_attn.ops.triton.rotary.apply_rotary.

    ``x``: (batch, seqlen, nheads, headdim); ``cos``/``sin``: (seqlen_ro,
    rotary_dim / 2).  Non-interleaved (NeoX) rotation applied to the first
    ``rotary_dim`` features, the rest passed through.  The production kernel
    computes in fp32 and rounds once when writing back to x.dtype.
    """
    rotary_dim = 2 * cos.shape[-1]
    seqlen = x.shape[1]
    cos = cos[:seqlen].to(device=x.device, dtype=torch.float32)
    sin = sin[:seqlen].to(device=x.device, dtype=torch.float32)
    # (seqlen, rot/2) -> (1, seqlen, 1, rot): same cos/sin for both halves.
    cos = torch.cat([cos, cos], dim=-1)[None, :, None, :]
    sin = torch.cat([sin, sin], dim=-1)[None, :, None, :]
    x_rot = x[..., :rotary_dim].float()
    out = (x_rot * cos + _rotate_half(x_rot) * sin).to(x.dtype)
    if rotary_dim < x.shape[-1]:
        out = torch.cat([out, x[..., rotary_dim:]], dim=-1)
    return out


class VisionAttention(nn.Module):
    """Multi-head attention for vision encoder (Qwen2-VL / Qwen2.5-VL / Qwen3-VL).

    All heads are attention heads (no GQA). Uses full (non-causal) attention.
    """

    def __init__(self, embed_dim: int, num_heads: int, projection_size: int | None = None):
        super().__init__()
        if projection_size is None:
            projection_size = embed_dim
        self.head_dim = projection_size // num_heads
        self.num_heads = num_heads

        self.qkv = QKVParallelLinear(
            embed_dim, self.head_dim, num_heads, num_heads, bias=True,
        )
        self.proj = RowParallelLinear(projection_size, embed_dim, bias=True)
        self.attn = FlashAttnPrefill(self.num_heads, self.num_heads, self.head_dim)

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        seq_len, batch_size, _ = x.shape
        qkv = self.qkv(x)

        q_size = self.num_heads * self.head_dim
        q, k, v = qkv.split([q_size, q_size, q_size], dim=-1)
        q = q.view(seq_len, batch_size, self.num_heads, self.head_dim)
        k = k.view(seq_len, batch_size, self.num_heads, self.head_dim)
        v = v.view(seq_len, batch_size, self.num_heads, self.head_dim)

        # Transpose to (batch, seq, heads, dim)
        q = q.transpose(0, 1).contiguous()
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()

        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            qk = torch.cat([q, k], dim=0)
            qk = apply_rotary(qk, rotary_pos_emb_cos, rotary_pos_emb_sin)
            q, k = qk.chunk(2, dim=0)

        # Flatten batch dim for varlen
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_heads, self.head_dim)
        v = v.reshape(-1, self.num_heads, self.head_dim)

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        out = self.attn(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            softmax_scale=self.head_dim ** -0.5,
            causal=False,
        )

        out = out.view(seq_len, batch_size, -1)
        return self.proj(out)


# Inlined from tasks/reference/L2/vision_mlp.py


class VisionMLP(nn.Module):
    """Vision encoder MLP with configurable activation.

    Qwen2-VL uses QuickGELU (default); Qwen3-VL uses F.silu.
    """

    def __init__(self, in_features: int, hidden_features: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 bias: bool = True):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.fc2 = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_hidden_dim: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim, eps=norm_eps)
        self.norm2 = LayerNorm(embed_dim, eps=norm_eps)
        self.attn = VisionAttention(embed_dim, num_heads)
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim, act_fn=act_fn)

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), cu_seqlens,
            rotary_pos_emb_cos, rotary_pos_emb_sin,
            max_seqlen,
        )
        x = x + self.mlp(self.norm2(x))
        return x
