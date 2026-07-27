from __future__ import annotations

from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


try:
    from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
    from fastkernels.tasks.baseline.L2.oasis_mlp import OasisMLP
    from fastkernels.tasks.baseline.L2.oasis_vae_attention import OasisVAEAttention
except Exception:
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
            return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(orig_dtype)

    class Matmul(nn.Module):
        def forward(self, input, weight, bias=None):
            return F.linear(input, weight, bias)

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
            return F.gelu(x, approximate=self.approximate)

    class OasisMLP(nn.Module):
        def __init__(
            self,
            in_features: int,
            hidden_features: int | None = None,
            out_features: int | None = None,
            *,
            approximate_tanh: bool = False,
        ):
            super().__init__()
            hidden_features = hidden_features or in_features
            out_features = out_features or in_features
            self.fc1 = Linear(in_features, hidden_features, bias=True)
            self.act = GELU(approximate="tanh" if approximate_tanh else "none")
            self.fc2 = Linear(hidden_features, out_features, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.act(self.fc1(x)))

    class DenseAttention(nn.Module):
        def __init__(self, backend: str = "auto"):
            super().__init__()
            del backend

        def forward(self, query, key, value, softmax_scale=None, causal=False, attn_mask=None):
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=causal,
                scale=softmax_scale,
            )
            return out.permute(0, 2, 1, 3)

    def oasis_rotate_half(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)

    def oasis_apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dtype = t.dtype
        rot_dim = freqs.shape[-1]
        t_middle = t[..., :rot_dim]
        t_right = t[..., rot_dim:]
        t_transformed = (t_middle * freqs.cos()) + (oasis_rotate_half(t_middle) * freqs.sin())
        return torch.cat((t_transformed, t_right), dim=-1).to(dtype)

    class OasisRotaryEmbedding(nn.Module):
        def __init__(
            self,
            dim: int,
            *,
            freqs_for: str = "lang",
            theta: float = 10000.0,
            max_freq: float = 10.0,
        ):
            super().__init__()
            self.dim = dim
            self.freqs_for = freqs_for
            if freqs_for == "lang":
                freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            elif freqs_for == "pixel":
                freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
            else:
                raise ValueError(f"unsupported rotary mode: {freqs_for}")
            self.freqs = nn.Parameter(freqs, requires_grad=False)
            self.register_buffer("dummy", torch.tensor(0), persistent=False)

        @property
        def device(self) -> torch.device:
            return self.dummy.device

        def _forward_freqs(self, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
            freqs = torch.einsum("..., f -> ... f", positions.to(freqs.dtype), freqs)
            return freqs.repeat_interleave(2, dim=-1)

        def forward(self, t: torch.Tensor, freqs: torch.Tensor, seq_len: int | None = None, offset: int = 0) -> torch.Tensor:
            del seq_len, offset
            return self._forward_freqs(t, freqs)

        def get_axial_freqs(self, *dims: int) -> torch.Tensor:
            all_freqs = []
            for index, dim in enumerate(dims):
                use_pixel = self.freqs_for == "pixel" and index >= len(dims) - 2
                if use_pixel:
                    pos = torch.linspace(-1, 1, steps=dim, device=self.device)
                else:
                    pos = torch.arange(dim, device=self.device)
                seq_freqs = self.forward(pos, self.freqs, seq_len=dim)
                axis = [None] * len(dims)
                axis[index] = slice(None)
                all_freqs.append(seq_freqs[(Ellipsis, *axis, slice(None))])
            all_freqs = torch.broadcast_tensors(*all_freqs)
            return torch.cat(all_freqs, dim=-1)

    class OasisVAEAttention(nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            frame_height: int,
            frame_width: int,
            *,
            qkv_bias: bool = False,
        ):
            super().__init__()
            self.num_heads = num_heads
            self.frame_height = frame_height
            self.frame_width = frame_width
            self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = Linear(dim, dim, bias=True)
            self.rotary = OasisRotaryEmbedding(
                dim=(dim // num_heads) // 4,
                freqs_for="pixel",
                max_freq=frame_height * frame_width,
            )
            self.register_buffer(
                "rotary_freqs",
                self.rotary.get_axial_freqs(frame_height, frame_width),
                persistent=False,
            )
            self.attn = DenseAttention(backend="sdpa")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
            k = k.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
            v = v.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
            q = oasis_apply_rotary_emb(self.rotary_freqs, q)
            k = oasis_apply_rotary_emb(self.rotary_freqs, k)
            seq_len = self.frame_height * self.frame_width
            q = q.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
            k = k.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
            v = v.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
            out = self.attn(q, k, v)
            out = out.reshape(bsz, seq_len, -1)
            return self.proj(out)


def _resolve_flash_attn():
    for mod_name, func_name in (
        ("flash_attn", "flash_attn_func"),
        ("flash_attn_interface", "flash_attn_func"),
        ("fa3_fwd_interface", "flash_attn_func"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[func_name])
            return getattr(mod, func_name)
        except Exception:
            pass
    return None


_FLASH_ATTN_FUNC = _resolve_flash_attn()
_FLASH_DISABLED = False


@triton.jit
def _ln_kernel(x_ptr, o_ptr, w_ptr, b_ptr, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    base = row * n_cols
    raw = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    x = raw.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (centered * rstd) * w + b
    tl.store(o_ptr + base + offs, y.to(raw.dtype), mask=mask)


@triton.jit
def _add_ln_kernel(r_ptr, a_ptr, s_ptr, n_ptr, w_ptr, b_ptr, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    base = row * n_cols
    r = tl.load(r_ptr + base + offs, mask=mask, other=0.0)
    a = tl.load(a_ptr + base + offs, mask=mask, other=0.0)
    summed = r + a
    tl.store(s_ptr + base + offs, summed, mask=mask)
    x = summed.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (centered * rstd) * w + b
    tl.store(n_ptr + base + offs, y.to(r.dtype), mask=mask)


@triton.jit
def _rotary_qk_kernel(qkv_ptr, cos_ptr, sin_ptr, total_pairs, seq_len: tl.constexpr, dim: tl.constexpr, num_heads: tl.constexpr, head_dim: tl.constexpr, rot_pairs: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_pairs
    pair = offs % rot_pairs
    t = offs // rot_pairs
    head = t % num_heads
    t = t // num_heads
    pos = t % seq_len
    batch = t // seq_len
    d0 = pair * 2
    base = (batch * seq_len + pos) * (dim * 3) + head * head_dim + d0
    c = tl.load(cos_ptr + pos * rot_pairs + pair, mask=mask, other=1.0).to(tl.float32)
    s = tl.load(sin_ptr + pos * rot_pairs + pair, mask=mask, other=0.0).to(tl.float32)

    q0 = tl.load(qkv_ptr + base, mask=mask, other=0.0).to(tl.float32)
    q1 = tl.load(qkv_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
    tl.store(qkv_ptr + base, q0 * c - q1 * s, mask=mask)
    tl.store(qkv_ptr + base + 1, q0 * s + q1 * c, mask=mask)

    kbase = base + dim
    k0 = tl.load(qkv_ptr + kbase, mask=mask, other=0.0).to(tl.float32)
    k1 = tl.load(qkv_ptr + kbase + 1, mask=mask, other=0.0).to(tl.float32)
    tl.store(qkv_ptr + kbase, k0 * c - k1 * s, mask=mask)
    tl.store(qkv_ptr + kbase + 1, k0 * s + k1 * c, mask=mask)


def _triton_block(n: int) -> int:
    return triton.next_power_of_2(n)


def _num_warps(block: int) -> int:
    return max(1, min(8, block // 256))


def _run_ln(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    if not x.is_cuda:
        return F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps).to(x.dtype)
    shape = x.shape
    n_cols = shape[-1]
    flat = x.contiguous().view(-1, n_cols)
    out = torch.empty_like(flat)
    block = _triton_block(n_cols)
    _ln_kernel[(flat.shape[0],)](
        flat,
        out,
        weight,
        bias,
        n_cols,
        eps,
        BLOCK=block,
        num_warps=_num_warps(block),
    )
    return out.view(shape)


def _run_add_ln(res: torch.Tensor, add: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not res.is_cuda:
        summed = res + add
        normed = F.layer_norm(summed.float(), (summed.shape[-1],), weight.float(), bias.float(), eps).to(summed.dtype)
        return summed, normed
    shape = res.shape
    n_cols = shape[-1]
    r = res.contiguous().view(-1, n_cols)
    a = add.contiguous().view(-1, n_cols)
    summed = torch.empty_like(r)
    normed = torch.empty_like(r)
    block = _triton_block(n_cols)
    _add_ln_kernel[(r.shape[0],)](
        r,
        a,
        summed,
        normed,
        weight,
        bias,
        n_cols,
        eps,
        BLOCK=block,
        num_warps=_num_warps(block),
    )
    return summed.view(shape), normed.view(shape)


def _run_rotary_qk(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, batch: int, seq_len: int, dim: int, num_heads: int, head_dim: int, rot_dim: int) -> None:
    rot_pairs = rot_dim // 2
    total_pairs = batch * seq_len * num_heads * rot_pairs
    _rotary_qk_kernel[(triton.cdiv(total_pairs, 256),)](
        qkv,
        cos,
        sin,
        total_pairs,
        seq_len,
        dim,
        num_heads,
        head_dim,
        rot_pairs,
        BLOCK=256,
        num_warps=4,
    )


def _flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    global _FLASH_DISABLED
    if _FLASH_ATTN_FUNC is None or _FLASH_DISABLED or q.dtype == torch.float32:
        return None
    try:
        out = _FLASH_ATTN_FUNC(q, k, v, dropout_p=0.0, causal=False)
        if isinstance(out, tuple):
            out = out[0]
        return out
    except Exception:
        _FLASH_DISABLED = True
        return None


class OasisVAEAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = OasisVAEAttention(
            dim,
            num_heads,
            frame_height,
            frame_width,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.mlp = OasisMLP(dim, hidden_features=int(dim * mlp_ratio), approximate_tanh=False)
        self._dim = dim
        self._num_heads = num_heads
        self._head_dim = dim // num_heads
        self._frame_height = frame_height
        self._frame_width = frame_width
        self._seq_len = frame_height * frame_width
        rotary_freqs = self.attn.rotary_freqs
        rot_dim = rotary_freqs.shape[-1]
        self._rot_dim = rot_dim
        pair_freqs = rotary_freqs.reshape(self._seq_len, rot_dim)[:, 0::2]
        self.register_buffer("_rot_cos_pair", pair_freqs.cos().contiguous(), persistent=False)
        self.register_buffer("_rot_sin_pair", pair_freqs.sin().contiguous(), persistent=False)

    def _baseline_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not x.is_cuda) or x.shape[0] <= 8:
            return self._baseline_forward(x)

        batch = x.shape[0]
        seq_len = self._seq_len
        dim = self._dim
        num_heads = self._num_heads
        head_dim = self._head_dim

        normed = _run_ln(x, self.norm1.weight, self.norm1.bias, 1e-6)
        qkv = F.linear(normed, self.attn.qkv.weight, self.attn.qkv.bias)
        _run_rotary_qk(
            qkv,
            self._rot_cos_pair,
            self._rot_sin_pair,
            batch,
            seq_len,
            dim,
            num_heads,
            head_dim,
            self._rot_dim,
        )

        q = qkv[..., :dim].reshape(batch, seq_len, num_heads, head_dim)
        k = qkv[..., dim:dim * 2].reshape(batch, seq_len, num_heads, head_dim)
        v = qkv[..., dim * 2:].reshape(batch, seq_len, num_heads, head_dim)

        out = _flash_attention(q, k, v)
        if out is None:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

        attn_out = F.linear(out.reshape(batch, seq_len, dim), self.attn.proj.weight, self.attn.proj.bias)
        x, normed = _run_add_ln(x, attn_out, self.norm2.weight, self.norm2.bias, 1e-6)
        hidden = F.gelu(F.linear(normed, self.mlp.fc1.weight, self.mlp.fc1.bias), approximate="none")
        hidden = F.linear(hidden, self.mlp.fc2.weight, self.mlp.fc2.bias)
        return hidden.add_(x)
