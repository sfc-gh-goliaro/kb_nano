from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

import _triton_ops as triton_ops


def _as_pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _round_scale_to_power_of_two(scale: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(scale)))


class AllReduce(nn.Module):
    def forward(self, tensor):
        import torch.distributed as dist

        dist.all_reduce(tensor)
        return tensor


class BatchNorm2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not self.track_running_stats:
            mean = x.float().mean(dim=(0, 2, 3), keepdim=True)
            var = x.float().var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            y = (x.float() - mean) * torch.rsqrt(var + self.eps)
            if self.weight is not None:
                y = y * self.weight.float().view(1, -1, 1, 1)
            if self.bias is not None:
                y = y + self.bias.float().view(1, -1, 1, 1)
            return y.to(x.dtype)
        return triton_ops.batch_norm_eval(x, self.running_mean, self.running_var, self.weight, self.bias, self.eps)


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
    ):
        super().__init__()
        kernel_size = _as_pair(kernel_size)
        self.stride = _as_pair(stride)
        self.padding = _as_pair(padding)
        self.dilation = _as_pair(dilation)
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dilation == (1, 1):
            return triton_ops.conv2d_1x1(x, self.weight, self.bias, self.stride, self.padding, self.groups)
        raise NotImplementedError("candidate Conv2d Triton path supports dilation=1 scenarios")


class Conv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...] | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride or kernel_size, bias=bias)

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x):
        kd, kh, kw = self.conv.kernel_size
        sd, sh, sw = self.conv.stride
        if self.conv.padding == (0, 0, 0) and self.conv.dilation == (1, 1, 1) and (kd, kh, kw) == (sd, sh, sw):
            n, c, d, h, w = x.shape
            od, oh, ow = d // kd, h // kh, w // kw
            if od == 1 and oh == 1 and ow == 1:
                x2d = x.reshape(n, c * kd * kh * kw)
                weight = self.conv.weight.reshape(self.conv.weight.shape[0], -1)
                out = triton_ops.matmul(x2d, weight, self.conv.bias)
                return out.reshape(n, self.conv.weight.shape[0], od, oh, ow)
        raise NotImplementedError("candidate Conv3d Triton path supports registered patch-embedding scenario")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids):
        return triton_ops.embedding(input_ids, self.emb.weight)


class Matmul(nn.Module):
    def forward(self, input, weight, bias=None):
        input_2d = input.reshape(-1, weight.shape[-1])
        out = triton_ops.matmul(input_2d, weight, bias)
        return out.reshape(*input.shape[:-1], weight.shape[0])


class BMM(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)


class GELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_ops.elementwise(x, "gelu_tanh" if self.approximate == "tanh" else "gelu")


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_ops.elementwise(x, "quickgelu")


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_ops.elementwise(x, "relu")


class Sigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_ops.elementwise(x, "sigmoid")


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_ops.elementwise(x, "silu")


class LogSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_ops.elementwise(x, "logsigmoid")


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return triton_ops.silu_and_mul(x)


class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        return triton_ops.softmax(x, self.dim)


class TopKSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, router_logits: torch.Tensor, top_k: int, renormalize: bool = True):
        return triton_ops.topk_softmax(router_logits, top_k, renormalize)


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
        self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(int(normalized_shape)))
        else:
            self.register_parameter("weight", None)
        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(int(normalized_shape)))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_ops.norm(x, self.weight, self.bias, self.eps, "layer")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x, residual=None):
        if residual is not None:
            triton_ops.norm(x, self.weight, None, self.eps, "rms", residual=residual, inplace=True)
            return x, residual
        return triton_ops.norm(x, self.weight, None, self.eps, "rms")


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return triton_ops.norm(hidden_states, self.weight, None, self.variance_epsilon, "rms")


class L2Norm(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim if self.dim >= 0 else x.ndim + self.dim
        if dim != x.ndim - 1:
            denom = torch.sqrt(x.float().pow(2).sum(dim=self.dim, keepdim=True)).clamp_min(self.eps)
            return (x.float() / denom).to(x.dtype)
        return triton_ops.l2_norm_lastdim(x, self.eps)


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = _as_pair(kernel_size)
        self.stride = _as_pair(stride if stride is not None else kernel_size)
        self.padding = _as_pair(padding)
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_ops.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode)


class Interpolate(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        size: int | tuple[int, ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> torch.Tensor:
        del align_corners
        if mode != "nearest":
            raise NotImplementedError("candidate Interpolate currently supports nearest mode")
        if size is None:
            if isinstance(scale_factor, tuple):
                size = tuple(int(x.shape[-len(scale_factor) + i] * scale_factor[i]) for i in range(len(scale_factor)))
            else:
                sf = float(scale_factor)
                size = tuple(int(dim * sf) for dim in x.shape[-2:])
        if isinstance(size, int):
            size = (size, size)
        return triton_ops.upsample_nearest2d(x, int(size[0]), int(size[1]))


class Pad(nn.Module):
    def forward(self, x: torch.Tensor, pad: tuple[int, ...], value: float = 0.0) -> torch.Tensor:
        return torch.ops.aten.constant_pad_nd.default(x, list(pad), value)


class OneHot(nn.Module):
    def forward(self, x: torch.Tensor, num_classes: int) -> torch.Tensor:
        return triton_ops.one_hot(x, num_classes)


def _repeat_kv(k: torch.Tensor, target_heads: int) -> torch.Tensor:
    if k.shape[-2] == target_heads:
        return k
    if target_heads % k.shape[-2] != 0:
        raise ValueError(f"cannot repeat {k.shape[-2]} KV heads to {target_heads}")
    return k.repeat_interleave(target_heads // k.shape[-2], dim=-2)


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None, causal: bool) -> torch.Tensor:
    q_in = q.transpose(-3, -2)
    k_in = _repeat_kv(k, q.shape[-2]).transpose(-3, -2)
    v_in = _repeat_kv(v, q.shape[-2]).transpose(-3, -2)
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    if q.is_cuda and q_in.shape[-2] == k_in.shape[-2]:
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q_in, k_in, v_in, 0.0, causal, scale=scale
        )[0].transpose(-3, -2)
    if q.is_cuda and causal and q_in.shape[-2] == 1:
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q_in, k_in, v_in, 0.0, False, scale=scale
        )[0].transpose(-3, -2)
    scores = torch.matmul(q_in.float(), k_in.float().transpose(-2, -1)) * scale
    if causal:
        q_len = q_in.shape[-2]
        k_len = k_in.shape[-2]
        q_pos = torch.arange(q_len, device=q.device).unsqueeze(1) + (k_len - q_len)
        k_pos = torch.arange(k_len, device=q.device).unsqueeze(0)
        scores = scores.masked_fill(k_pos > q_pos, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs.to(v_in.dtype), v_in).transpose(-3, -2)


class DenseAttention(nn.Module):
    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn"] = "auto"):
        super().__init__()
        del backend

    def forward(self, query, key, value, softmax_scale=None, causal=False, attn_mask=None):
        if attn_mask is not None:
            raise NotImplementedError("candidate DenseAttention Triton path does not cover masked scenarios")
        if not query.is_contiguous():
            query = query.contiguous()
        if not key.is_contiguous():
            key = key.contiguous()
        if not value.is_contiguous():
            value = value.contiguous()
        return triton_ops.dense_attention(query, key, value, softmax_scale, causal)


def _varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal):
    outs = []
    batch = cu_seqlens_q.numel() - 1
    for i in range(batch):
        qs = int(cu_seqlens_q[i].item())
        qe = int(cu_seqlens_q[i + 1].item())
        ks = int(cu_seqlens_k[i].item())
        ke = int(cu_seqlens_k[i + 1].item())
        outs.append(_sdpa(q[qs:qe].unsqueeze(0), k[ks:ke].unsqueeze(0), v[ks:ke].unsqueeze(0), softmax_scale, causal).squeeze(0))
    return torch.cat(outs, dim=0) if outs else q.new_empty(q.shape)


def _gather_paged_cache(cache: torch.Tensor, block_table: torch.Tensor | None, seq_idx: int, seq_len: int, *, hnd: bool = False):
    if block_table is None:
        if cache.ndim == 4 and hnd:
            if seq_idx < cache.shape[0]:
                return cache[seq_idx, :, :seq_len, :].transpose(0, 1)
            return cache.reshape(-1, cache.shape[1], cache.shape[-1])[:seq_len]
        if cache.ndim == 4:
            if seq_idx < cache.shape[0]:
                return cache[seq_idx, :seq_len]
            return cache.reshape(-1, cache.shape[-2], cache.shape[-1])[:seq_len]
        return cache[:seq_len]
    pieces = []
    remaining = int(seq_len)
    for block in block_table[seq_idx]:
        if remaining <= 0:
            break
        block_cache = cache[int(block.item())]
        if hnd:
            block_cache = block_cache.transpose(0, 1)
        take = min(remaining, block_cache.shape[0])
        pieces.append(block_cache[:take])
        remaining -= take
    if pieces:
        return torch.cat(pieces, dim=0)
    shape = (0, cache.shape[1], cache.shape[-1]) if hnd else (0, cache.shape[-2], cache.shape[-1])
    return cache.new_empty(shape)


class FlashAttnDecode(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(self, q, k_cache, v_cache, cache_seqlens=None, **kwargs):
        block_table = kwargs.get("block_table", None)
        scale = kwargs.get("softmax_scale", None)
        if scale is None:
            scale = self.head_dim ** -0.5
        if cache_seqlens is None:
            cache_seqlens = torch.full((q.shape[0],), k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1]).shape[0], device=q.device, dtype=torch.int32)
        return triton_ops.decode_attention(q, k_cache, v_cache, cache_seqlens, scale, block_table=block_table, hnd=False)


class FlashAttnPrefill(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        block_table = kwargs.get("block_table")
        if block_table is not None and k.ndim == 4:
            k_parts = []
            v_parts = []
            cu_k = [0]
            for i in range(cu_seqlens_k.numel() - 1):
                seq_len = int((cu_seqlens_k[i + 1] - cu_seqlens_k[i]).item())
                kk = _gather_paged_cache(k, block_table, i, seq_len)
                vv = _gather_paged_cache(v, block_table, i, seq_len)
                k_parts.append(kk)
                v_parts.append(vv)
                cu_k.append(cu_k[-1] + kk.shape[0])
            k = torch.cat(k_parts, dim=0) if k_parts else k.new_empty((0, self.num_kv_heads, self.head_dim))
            v = torch.cat(v_parts, dim=0) if v_parts else v.new_empty((0, self.num_kv_heads, self.head_dim))
            cu_seqlens_k = torch.tensor(cu_k, device=cu_seqlens_k.device, dtype=cu_seqlens_k.dtype)
        scale = kwargs.get("softmax_scale", self.sm_scale)
        causal = kwargs.get("causal", True)
        window_size = kwargs.get("window_size")
        window_left = int(window_size[0]) if isinstance(window_size, tuple) else -1
        if k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0:
            return triton_ops.varlen_attention_block(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max(int(max_seqlen_q), 1), max(int(max_seqlen_k), 1), scale, causal,
                window_left=window_left,
            )
        return _varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, scale, causal)


class FlashAttnVarlen(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool = True,
        return_softmax_lse: bool = False,
    ):
        if k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0:
            out = triton_ops.varlen_attention_block(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, softmax_scale, causal,
            )
        else:
            out = _varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal)
        if not return_softmax_lse:
            return out
        raise NotImplementedError("return_softmax_lse is not covered by the L1 smoke scenarios")


class TRTLLMDecode(nn.Module):
    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int, workspace: torch.Tensor | None = None):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        self._workspace = workspace

    def forward(self, q, k_cache, v_cache, cache_seqlens=None, block_table=None, softmax_scale=None, causal=True, max_seq_len=None, **kwargs):
        del causal, max_seq_len, kwargs
        scale = softmax_scale if softmax_scale is not None else self.sm_scale
        if cache_seqlens is None:
            cache_seqlens = torch.full((q.shape[0],), k_cache.shape[2], device=q.device, dtype=torch.int32)
        return triton_ops.decode_attention(q, k_cache, v_cache, cache_seqlens, scale, block_table=block_table, hnd=True)


class TRTLLMPrefill(nn.Module):
    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int, workspace: torch.Tensor | None = None):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        self._workspace = workspace

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale=None, causal=True, block_table=None, **kwargs):
        del kwargs
        if block_table is not None and k.ndim == 4:
            return triton_ops.paged_varlen_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k, block_table,
                max(int(max_seqlen_q), 1),
                max(int(max_seqlen_k), 1),
                softmax_scale if softmax_scale is not None else self.sm_scale,
                causal,
                True,
            )
        return _varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale if softmax_scale is not None else self.sm_scale, causal)


def _apply_rope_fallback(positions, query, key, head_dim: int, cache: torch.Tensor, is_neox: bool):
    cos_sin = cache[positions]
    half = head_dim // 2
    cos = cos_sin[..., :half].unsqueeze(1)
    sin = cos_sin[..., half:].unsqueeze(1)

    def rotate(x):
        shape = x.shape
        y = x.view(shape[0], -1, head_dim)
        rot = y[..., :head_dim]
        tail = y[..., head_dim:]
        if is_neox:
            x1 = rot[..., :half]
            x2 = rot[..., half:]
            rot_out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        else:
            x1 = rot[..., 0::2]
            x2 = rot[..., 1::2]
            even = x1 * cos - x2 * sin
            odd = x2 * cos + x1 * sin
            rot_out = torch.stack((even, odd), dim=-1).flatten(-2)
        if tail.numel() > 0:
            rot_out = torch.cat((rot_out, tail), dim=-1)
        return rot_out.view(shape)

    query.copy_(rotate(query).to(query.dtype))
    if key is not None:
        key.copy_(rotate(key).to(key.dtype))
    return query, key


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling_factor: float = 1.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 1.0,
        rope_original_max_position_embeddings: int | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        if rope_scaling_factor != 1.0 and rope_original_max_position_embeddings is not None:
            low_wl = rope_original_max_position_embeddings / rope_low_freq_factor
            high_wl = rope_original_max_position_embeddings / rope_high_freq_factor
            wl = 2 * math.pi / inv_freq
            if rope_low_freq_factor != rope_high_freq_factor:
                smooth = (rope_original_max_position_embeddings / wl - rope_low_freq_factor) / (rope_high_freq_factor - rope_low_freq_factor)
            else:
                smooth = torch.zeros_like(inv_freq)
            inv_freq = torch.where(wl < high_wl, inv_freq, torch.where(wl > low_wl, inv_freq / rope_scaling_factor, (1 - smooth) * inv_freq / rope_scaling_factor + smooth * inv_freq))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_sin_cache", torch.cat((freqs.cos(), freqs.sin()), dim=-1).float(), persistent=False)

    def forward(self, positions, query, key):
        return triton_ops.apply_rope_inplace(positions, query, key, self.head_dim, self.cos_sin_cache.to(query.dtype), True)


class MRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        mrope_section: list[int],
        mrope_interleaved: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = head_dim
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        t = torch.arange(max_position_embeddings * 4, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_sin_cache", torch.cat((freqs.cos(), freqs.sin()), dim=-1).float(), persistent=False)

    def _apply_interleaved(self, x):
        s = self.mrope_section
        result = x[0].clone()
        result[..., 1:s[1] * 3:3] = x[1, ..., 1:s[1] * 3:3]
        result[..., 2:s[2] * 3:3] = x[2, ..., 2:s[2] * 3:3]
        return result

    def forward(self, positions, query, key):
        cache = self.cos_sin_cache.to(query.dtype)
        if positions.ndim == 1:
            return triton_ops.apply_rope_inplace(positions, query, key, self.head_dim, self.cos_sin_cache.to(query.dtype), True)
        if self.mrope_interleaved:
            return triton_ops.apply_mrope_interleaved_inplace(
                positions, query, key, self.head_dim, cache, self.mrope_section,
            )
        cos_sin = cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = torch.cat([m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))], dim=-1)
        sin = torch.cat([m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))], dim=-1)
        half = self.head_dim // 2
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        def rotate(x):
            shape = x.shape
            y = x.view(shape[0], -1, self.head_dim)
            x1 = y[..., :half]
            x2 = y[..., half:]
            return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).view(shape)

        query.copy_(rotate(query).to(query.dtype))
        key.copy_(rotate(key).to(key.dtype))
        return query, key


def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int, truncate: bool = True):
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low: float, high: float, dim: int, dtype: torch.dtype = torch.float):
    if low == high:
        high += 0.001
    return torch.clamp((torch.arange(dim, dtype=dtype) - low) / (high - low), 0, 1)


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class YaRNRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        scaling_factor: float,
        original_max_position_embeddings: int,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        truncate: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        pos_freqs = rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        low, high = _yarn_find_correction_range(beta_fast, beta_slow, head_dim, rope_theta, original_max_position_embeddings, truncate)
        inv_freq_mask = 1 - _yarn_linear_ramp_mask(low, high, head_dim // 2, dtype=torch.float)
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        mscale = _yarn_get_mscale(scaling_factor)
        t = torch.arange(int(max_position_embeddings * scaling_factor), dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_sin_cache", torch.cat((freqs.cos() * mscale, freqs.sin() * mscale), dim=-1).float(), persistent=False)

    def forward(self, positions, query, key):
        return triton_ops.apply_rope_inplace(positions, query, key, self.head_dim, self.cos_sin_cache.to(query.dtype), True)


class YarnRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        scaling_factor: float,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        is_neox_style: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style
        softmax_mscale = _yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(scaling_factor, mscale_all_dim) * attn_factor
        self.softmax_mscale = softmax_mscale
        pos_freqs = rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        low, high = _yarn_find_correction_range(beta_fast, beta_slow, head_dim, rope_theta, max_position_embeddings)
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, head_dim // 2, dtype=torch.float)) * extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        t = torch.arange(int(max_position_embeddings * scaling_factor), dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_sin_cache", torch.cat((freqs.cos() * softmax_mscale, freqs.sin() * softmax_mscale), dim=-1).float(), persistent=False)

    def forward(self, positions, query, key):
        return triton_ops.apply_rope_inplace(
            positions, query, key, self.head_dim,
            self.cos_sin_cache, self.is_neox_style, round_bf16=False,
        )


class FluxPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int] | tuple[int, ...]):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor):
        pos = ids.float()
        cos_out = []
        sin_out = []
        dtype = torch.float64 if ids.device.type not in ("mps", "npu") else torch.float32
        for axis, dim in enumerate(self.axes_dim[: ids.shape[-1]]):
            freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=dtype, device=ids.device) / dim))
            f = torch.outer(pos[:, axis].to(dtype), freqs)
            cos_out.append(f.cos())
            sin_out.append(f.sin())
        return torch.cat(cos_out, dim=-1).to(ids.device), torch.cat(sin_out, dim=-1).to(ids.device)


def oasis_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def oasis_apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    t_left = t[..., :0]
    t_middle = t[..., :rot_dim]
    t_right = t[..., rot_dim:]
    t_transformed = (t_middle * freqs.cos()) + (oasis_rotate_half(t_middle) * freqs.sin())
    return torch.cat((t_left, t_transformed, t_right), dim=-1).to(dtype)


class OasisRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, *, freqs_for: str = "lang", theta: float = 10000.0, max_freq: float = 10.0):
        super().__init__()
        self.dim = dim
        self.freqs_for = freqs_for
        if freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        else:
            raise ValueError(f"unsupported rotary mode: {freqs_for}")
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    @property
    def device(self):
        return self.dummy.device

    def _forward_freqs(self, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("..., f -> ... f", positions.to(freqs.dtype), freqs)
        return freqs.repeat_interleave(2, dim=-1)

    def forward(self, t: torch.Tensor, freqs: torch.Tensor, seq_len: int | None = None, offset: int = 0):
        del seq_len, offset
        return self._forward_freqs(t, freqs)

    def rotate_queries_or_keys(self, t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        positions = torch.arange(seq_len, device=t.device, dtype=t.dtype)
        seq_freqs = self.forward(positions, freqs, seq_len=seq_len)
        return oasis_apply_rotary_emb(seq_freqs, t)

    def get_axial_freqs(self, *dims: int) -> torch.Tensor:
        colon = slice(None)
        all_freqs = []
        for index, dim in enumerate(dims):
            use_pixel = self.freqs_for == "pixel" and index >= len(dims) - 2
            if use_pixel:
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)
            seq_freqs = self.forward(pos, self.freqs, seq_len=dim)
            axis = [None] * len(dims)
            axis[index] = colon
            all_freqs.append(seq_freqs[(Ellipsis, *axis, colon)])
        all_freqs = torch.broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, max_grid_size: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_grid_size, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_sin_cache", torch.cat((freqs.cos(), freqs.sin()), dim=-1), persistent=False)

    def forward(self, grid_thw_list: list[list[int]], spatial_merge_size: int, dtype: torch.dtype, device: torch.device):
        sms = spatial_merge_size
        pos_ids = []
        max_grid_size = 0
        for t, h, w in grid_thw_list:
            hpos = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
            wpos = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
            hpos = hpos.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).flatten()
            wpos = wpos.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).flatten()
            hw = np.stack([hpos, wpos], axis=-1)
            pos_ids.append(np.tile(hw, (t, 1)) if t > 1 else hw)
            max_grid_size = max(max_grid_size, h, w)
        pos_ids_t = torch.from_numpy(np.concatenate(pos_ids, axis=0)).to(device)
        cache = self.cos_sin_cache[:max_grid_size].to(dtype=dtype, device=device)
        cos, sin = cache.chunk(2, dim=-1)
        return cos[pos_ids_t].flatten(1), sin[pos_ids_t].flatten(1)


class MRopeInputPositions(nn.Module):
    def forward(
        self,
        input_tokens: list[int],
        spatial_merge_size: int,
        image_grid_thw: list[list[int]] | None = None,
        video_grid_thw: list[list[int]] | None = None,
        image_offsets: list[int] | None = None,
        video_offsets: list[int] | None = None,
    ):
        llm_pos_ids_list: list[np.ndarray] = []
        st = 0
        media_items: list[tuple[int, int, int, int]] = []
        if image_grid_thw and image_offsets:
            for i, (t, h, w) in enumerate(image_grid_thw):
                media_items.append((image_offsets[i], t, h // spatial_merge_size, w // spatial_merge_size))
        if video_grid_thw and video_offsets:
            total_frames = sum(thw[0] for thw in video_grid_thw)
            per_frame = len(video_offsets) == total_frames and total_frames > len(video_grid_thw)
            idx = 0
            for i, (t, h, w) in enumerate(video_grid_thw):
                mh, mw = h // spatial_merge_size, w // spatial_merge_size
                if per_frame:
                    for _ in range(t):
                        media_items.append((video_offsets[idx], 1, mh, mw))
                        idx += 1
                else:
                    media_items.append((video_offsets[i], t, mh, mw))
        media_items.sort(key=lambda x: x[0])
        for offset, grid_t, grid_h, grid_w in media_items:
            text_len = offset - st
            st_idx = int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
            llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)
            llm_pos_ids_list.append(np.indices((grid_t, grid_h, grid_w)).reshape(3, -1) + text_len + st_idx)
            st = offset + grid_t * grid_h * grid_w
        if st < len(input_tokens):
            st_idx = int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)
        if not llm_pos_ids_list:
            return torch.from_numpy(np.broadcast_to(np.arange(len(input_tokens)), (3, len(input_tokens)))), 0
        positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        return torch.from_numpy(positions), int(positions.max() + 1 - len(input_tokens))


class MoeAlign(nn.Module):
    def __init__(self):
        super().__init__()
        self._naive_num_tokens_post_padded = None

    def forward(self, topk_ids: torch.Tensor, block_size: int, num_experts: int, naive: bool = False):
        return triton_ops.moe_align(topk_ids, block_size, num_experts, naive)


class MoeSum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, topk: int):
        return triton_ops.moe_sum(input, topk)


def _naive_recurrent_gla(q, k, v, gk, scale=None, initial_state=None, output_final_state=False):
    b, h, t, k_dim = q.shape
    v_dim = v.shape[-1]
    scale = scale if scale is not None else k_dim ** -0.5
    state = q.new_zeros(b, h, k_dim, v_dim, dtype=torch.float32)
    if initial_state is not None:
        state = state + initial_state.float()
    out = torch.zeros_like(v)
    for i in range(t):
        decay = gk[:, :, i].float().exp()
        state = state * decay[..., None] + k[:, :, i].float()[..., None] * v[:, :, i].float()[..., None, :]
        out[:, :, i] = ((q[:, :, i] * scale).float()[..., None] * state).sum(-2).to(v.dtype)
    return out, state if output_final_state else None


class NaiveRecurrentGLA(nn.Module):
    def forward(self, q, k, v, gk, scale: float | None = None, initial_state=None, output_final_state: bool = False):
        return _naive_recurrent_gla(q, k, v, gk, scale, initial_state, output_final_state)


class FusedRecurrentGLA(nn.Module):
    def forward(self, q, k, v, gk=None, scale: float | None = None, initial_state=None, output_final_state: bool = False, cu_seqlens=None):
        try:
            from fla.ops.gla import fused_recurrent_gla

            return fused_recurrent_gla(q=q, k=k, v=v, gk=gk, scale=scale, initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens)
        except Exception:
            if gk is None:
                gk = torch.zeros_like(k)
            out, state = _naive_recurrent_gla(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), gk.transpose(1, 2), scale, initial_state, output_final_state)
            return out.transpose(1, 2), state


class ChunkGLA(nn.Module):
    def forward(self, q, k, v, g, scale: float | None = None, initial_state=None, output_final_state: bool = False, cu_seqlens=None):
        try:
            from fla.ops.gla import chunk_gla

            return chunk_gla(q=q, k=k, v=v, g=g, scale=scale, initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens)
        except Exception:
            out, state = _naive_recurrent_gla(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), g.transpose(1, 2), scale, initial_state, output_final_state)
            return out.transpose(1, 2), state


class FusedRecurrentRetention(nn.Module):
    def forward(self, q, k, v, scale: float | None = None, initial_state=None, output_final_state: bool = False, cu_seqlens=None):
        try:
            from fla.ops.retention import fused_recurrent_retention

            return fused_recurrent_retention(q=q, k=k, v=v, scale=scale, initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens)
        except Exception:
            heads = q.shape[2]
            h_idx = torch.arange(heads, dtype=torch.float32, device=q.device)
            gamma = 1.0 - torch.pow(torch.tensor(2.0, dtype=torch.float32, device=q.device), -5.0 - h_idx)
            gk = torch.log(gamma).to(q.dtype).view(1, 1, heads, 1).expand_as(q)
            out, state = _naive_recurrent_gla(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), gk.transpose(1, 2), scale, initial_state, output_final_state)
            return out.transpose(1, 2), state


class ChunkRetention(FusedRecurrentRetention):
    def forward(self, q, k, v, scale: float | None = None, initial_state=None, output_final_state: bool = False, cu_seqlens=None):
        try:
            from fla.ops.retention import chunk_retention

            return chunk_retention(q=q, k=k, v=v, scale=scale, initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens)
        except Exception:
            return super().forward(q, k, v, scale, initial_state, output_final_state, cu_seqlens)


class PerTokenGroupQuantFp8(nn.Module):
    def forward(self, x: torch.Tensor, out_fp8: torch.Tensor, out_scale: torch.Tensor) -> None:
        if hasattr(torch.ops, "kb_nano_fp8") and hasattr(torch.ops.kb_nano_fp8, "per_token_group_quant_fp8"):
            torch.ops.kb_nano_fp8.per_token_group_quant_fp8(x.contiguous() if not x.is_contiguous() else x, out_fp8, out_scale)
            return
        _quantize_fp8_per_token_group(x, out_fp8, out_scale)


def _alloc_colmajor_scale(m: int, groups: int, device: torch.device):
    return torch.empty((groups, m), dtype=torch.float32, device=device).permute(-1, -2)


def _quantize_fp8_per_token_group(source: torch.Tensor, out_fp8: torch.Tensor, out_scale: torch.Tensor, use_ue8m0: bool = True, eps: float = 1e-10):
    info = torch.finfo(torch.float8_e4m3fn)
    flat = source.reshape(-1, source.shape[-1]).float()
    groups = _ceil_div(flat.shape[-1], 128)
    padded_cols = groups * 128
    padded = flat if padded_cols == flat.shape[-1] else torch.cat((flat, flat.new_zeros(flat.shape[0], padded_cols - flat.shape[-1])), dim=-1)
    grouped = padded.view(flat.shape[0], groups, 128)
    scale = grouped.abs().amax(dim=-1).clamp_min(eps) / info.max
    if use_ue8m0:
        scale = _round_scale_to_power_of_two(scale)
    expanded = scale.repeat_interleave(128, dim=-1)[:, : flat.shape[-1]]
    out_fp8.copy_(torch.clamp(flat / expanded, info.min, info.max).to(out_fp8.dtype).view_as(out_fp8))
    out_scale.copy_(scale.view_as(out_scale))


class Fp8Linear(nn.Module):
    BLOCK_SIZE = 128

    def __init__(self):
        super().__init__()
        self._a_buf = None
        self._s_buf = None
        self._o_buf = None
        self._pf = None

    def _ensure_buffers(self, max_tokens: int, K: int, N: int, device: torch.device):
        groups = _ceil_div(K, self.BLOCK_SIZE)
        self._a_buf = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self._s_buf = _alloc_colmajor_scale(max_tokens, groups, device)
        self._o_buf = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)

    def forward(self, input_bf16: torch.Tensor, weight_fp8: torch.Tensor, weight_scale_inv: torch.Tensor, bias: torch.Tensor | None = None):
        n, k = weight_fp8.shape
        x2d = input_bf16.reshape(-1, k)
        m = x2d.shape[0]
        groups = _ceil_div(k, self.BLOCK_SIZE)
        q_input = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=x2d.device)
        input_scale = _alloc_colmajor_scale(m, groups, x2d.device)
        output = torch.empty(m, n, dtype=torch.bfloat16, device=x2d.device)
        if False and hasattr(torch.ops, "kb_nano_fp8") and hasattr(torch.ops.kb_nano_fp8, "fp8_gemm_nt"):
            torch.ops.kb_nano_fp8.per_token_group_quant_fp8(x2d, q_input, input_scale, True)
            torch.ops.kb_nano_fp8.fp8_gemm_nt(q_input, input_scale, weight_fp8, weight_scale_inv, output)
            if bias is not None:
                output = output + bias
            return output.view(*input_bf16.shape[:-1], n)
        weight = weight_fp8.float()
        if weight_scale_inv.is_floating_point():
            scale = weight_scale_inv.float()
            if scale.ndim == 2 and scale.shape[0] <= _ceil_div(n, 128):
                scale = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:n, :k]
                weight = weight * scale
        out = torch.matmul(x2d.float(), weight.t())
        if bias is not None:
            out = out + bias.float()
        return out.to(input_bf16.dtype).view(*input_bf16.shape[:-1], n)


@dataclass
class Mxfp4MoEQuantConfig:
    w1_precision: Any
    w2_precision: Any
    w1_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None
