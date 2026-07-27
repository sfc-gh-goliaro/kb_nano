from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


_FLASH_ATTN_VARLEN = None
_FLASH_ATTN_IMPORT_FAILED = False


def _tp_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _tp_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        create_scale: bool = True,
        create_offset: bool = True,
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.promote_fp32 = promote_fp32
        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)
        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _layer_norm(x, self.weight, self.bias, self.eps, self.normalized_shape)


class AllReduce(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor)
        return tensor


class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False, quant_config: dict | None = None):
        super().__init__()
        if quant_config is not None:
            raise NotImplementedError("fp8 quant_config is not supported by this VisionBlock")
        tp = _tp_size()
        assert output_size % tp == 0
        self.output_size_per_partition = output_size // tp
        self.use_fp8 = False
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size))
        self.weight.weight_loader = self._weight_loader
        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class QKVParallelLinear(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        bias: bool = False,
        quant_config: dict | None = None,
    ):
        super().__init__()
        if quant_config is not None:
            raise NotImplementedError("fp8 quant_config is not supported by this VisionBlock")
        tp = _tp_size()
        self.head_size = head_size
        self.num_heads = total_num_heads // tp
        if total_num_kv_heads % tp == 0:
            self.num_kv_heads = total_num_kv_heads // tp
            self._replicate_kv = False
        else:
            self.num_kv_heads = total_num_kv_heads
            self._replicate_kv = True
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
        self.use_fp8 = False
        self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
        self.weight.weight_loader = self._weight_loader
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
            src = loaded_weight.chunk(tp, 0)[rank]
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
            src = loaded_weight if self._replicate_kv else loaded_weight.chunk(tp, 0)[rank]
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            src = loaded_weight if self._replicate_kv else loaded_weight.chunk(tp, 0)[rank]
        param.data.narrow(0, shard_offset, shard_size).copy_(src)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: dict | None = None,
        reduce_results: bool = True,
    ):
        super().__init__()
        if quant_config is not None:
            raise NotImplementedError("fp8 quant_config is not supported by this VisionBlock")
        tp = _tp_size()
        assert input_size % tp == 0
        self.input_size_per_partition = input_size // tp
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.reduce_results = reduce_results
        self.use_fp8 = False
        self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition))
        self.weight.weight_loader = self._weight_loader
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)
        self.allreduce = AllReduce()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(1)
        param.data.copy_(loaded_weight.narrow(1, rank * shard, shard))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.reduce_results and self.tp_size > 1:
            y = self.allreduce(y)
        return y


@triton.jit
def _layer_norm_kernel(
    X,
    W,
    B,
    Y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.rsqrt(var + EPS)
    y = xc * rstd
    if HAS_WEIGHT:
        w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)
        y = y * w
    if HAS_BIAS:
        b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
        y = y + b
    tl.store(Y + row * N + offs, y, mask=mask)


@triton.jit
def _qkv_rotary_pack_kernel(
    QKV,
    Q,
    K,
    V,
    COS,
    SIN,
    TOTAL: tl.constexpr,
    NHEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, BLOCK_D)
    mask = (rm[:, None] < TOTAL) & (rd[None, :] < HEAD_DIM)
    width = NHEADS * HEAD_DIM
    q_base = rm[:, None] * (3 * width) + pid_h * HEAD_DIM + rd[None, :]
    k_base = q_base + width
    v_base = q_base + 2 * width
    qx = tl.load(QKV + q_base, mask=mask, other=0.0).to(tl.float32)
    kx = tl.load(QKV + k_base, mask=mask, other=0.0).to(tl.float32)
    vx = tl.load(QKV + v_base, mask=mask, other=0.0)
    rotary_dim: tl.constexpr = ROTARY_HALF * 2
    first = rd < ROTARY_HALF
    second = (rd >= ROTARY_HALF) & (rd < rotary_dim)
    cd = tl.where(first, rd, rd - ROTARY_HALF)
    cs_mask = (rm[:, None] < TOTAL) & (cd[None, :] < ROTARY_HALF) & (rd[None, :] < rotary_dim)
    cos = tl.load(COS + rm[:, None] * ROTARY_HALF + cd[None, :], mask=cs_mask, other=1.0).to(tl.float32)
    sin = tl.load(SIN + rm[:, None] * ROTARY_HALF + cd[None, :], mask=cs_mask, other=0.0).to(tl.float32)
    q_partner = tl.where(
        first[None, :],
        tl.load(QKV + q_base + ROTARY_HALF, mask=mask & first[None, :], other=0.0).to(tl.float32),
        tl.load(QKV + q_base - ROTARY_HALF, mask=mask & second[None, :], other=0.0).to(tl.float32),
    )
    k_partner = tl.where(
        first[None, :],
        tl.load(QKV + k_base + ROTARY_HALF, mask=mask & first[None, :], other=0.0).to(tl.float32),
        tl.load(QKV + k_base - ROTARY_HALF, mask=mask & second[None, :], other=0.0).to(tl.float32),
    )
    q_rot = tl.where(first[None, :], qx * cos - q_partner * sin, q_partner * sin + qx * cos)
    k_rot = tl.where(first[None, :], kx * cos - k_partner * sin, k_partner * sin + kx * cos)
    q_out = tl.where(rd[None, :] < rotary_dim, q_rot, qx)
    k_out = tl.where(rd[None, :] < rotary_dim, k_rot, kx)
    out_base = rm[:, None] * width + pid_h * HEAD_DIM + rd[None, :]
    tl.store(Q + out_base, q_out, mask=mask)
    tl.store(K + out_base, k_out, mask=mask)
    tl.store(V + out_base, vx, mask=mask)


@triton.jit
def _quickgelu_inplace_kernel(X, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = x / (1.0 + tl.exp(-1.702 * x))
    tl.store(X + offs, y, mask=mask)


@triton.jit
def _silu_inplace_kernel(X, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = x / (1.0 + tl.exp(-x))
    tl.store(X + offs, y, mask=mask)


def _layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    normalized_shape: tuple[int, ...],
) -> torch.Tensor:
    n = normalized_shape[0]
    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16) and n <= 4096 and x.shape[-1] == n:
        flat = x.reshape(-1, n)
        out = torch.empty_like(flat)
        block = triton.next_power_of_2(n)
        with torch.cuda.device(x.device.index if x.device.index is not None else torch.cuda.current_device()):
            _layer_norm_kernel[(flat.shape[0],)](
                flat,
                weight,
                bias,
                out,
                n,
                eps,
                weight is not None,
                bias is not None,
                BLOCK_N=block,
                num_warps=8,
            )
        return out.view_as(x)
    w = weight.float() if weight is not None and weight.dtype != torch.float32 else weight
    b = bias.float() if bias is not None and bias.dtype != torch.float32 else bias
    return F.layer_norm(x.float(), normalized_shape, w, b, eps).to(x.dtype)


def _flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
) -> torch.Tensor:
    global _FLASH_ATTN_VARLEN, _FLASH_ATTN_IMPORT_FAILED
    if _FLASH_ATTN_VARLEN is None and not _FLASH_ATTN_IMPORT_FAILED:
        try:
            from flash_attn import flash_attn_varlen_func

            _FLASH_ATTN_VARLEN = flash_attn_varlen_func
        except Exception:
            _FLASH_ATTN_IMPORT_FAILED = True
    if _FLASH_ATTN_VARLEN is not None:
        return _FLASH_ATTN_VARLEN(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            softmax_scale=softmax_scale,
            causal=False,
        )
    outs = []
    qh = q.transpose(0, 1).unsqueeze(0)
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)
    if cu_seqlens.numel() == 2:
        out = F.scaled_dot_product_attention(qh, kh, vh, scale=softmax_scale, is_causal=False)
        return out.squeeze(0).transpose(0, 1).contiguous()
    for i in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        out = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1).unsqueeze(0),
            k[start:end].transpose(0, 1).unsqueeze(0),
            v[start:end].transpose(0, 1).unsqueeze(0),
            scale=softmax_scale,
            is_causal=False,
        )
        outs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0).contiguous()


def _pack_qkv_rotary(
    qkv: torch.Tensor,
    nheads: int,
    head_dim: int,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = qkv.shape[0]
    if (
        qkv.is_cuda
        and qkv.is_contiguous()
        and cos is not None
        and sin is not None
        and cos.is_cuda
        and sin.is_cuda
        and cos.ndim == 2
        and sin.shape == cos.shape
        and cos.shape[0] >= total
        and cos.shape[1] * 2 <= head_dim
        and head_dim <= 256
    ):
        q = torch.empty((total, nheads, head_dim), device=qkv.device, dtype=qkv.dtype)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        block_d = triton.next_power_of_2(head_dim)
        with torch.cuda.device(qkv.device.index if qkv.device.index is not None else torch.cuda.current_device()):
            _qkv_rotary_pack_kernel[(triton.cdiv(total, 16), nheads)](
                qkv,
                q,
                k,
                v,
                cos.contiguous(),
                sin.contiguous(),
                total,
                nheads,
                head_dim,
                cos.shape[1],
                BLOCK_M=16,
                BLOCK_D=block_d,
                num_warps=4,
            )
        return q, k, v
    qkv = qkv.view(total, 3, nheads, head_dim)
    q = qkv[:, 0].contiguous()
    k = qkv[:, 1].contiguous()
    v = qkv[:, 2].contiguous()
    if cos is not None and sin is not None:
        q = _apply_rotary_torch(q, cos, sin)
        k = _apply_rotary_torch(k, cos, sin)
    return q, k, v


def _apply_rotary_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[: x.shape[0]].to(device=x.device, dtype=x.dtype)
    sin = sin[: x.shape[0]].to(device=x.device, dtype=x.dtype)
    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    half = rotary_dim // 2
    c = cos[:, None, :]
    s = sin[:, None, :]
    x0 = x_rot[..., :half]
    x1 = x_rot[..., half:]
    out = torch.cat((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1)
    if x_pass.numel():
        out = torch.cat((out, x_pass), dim=-1)
    return out.contiguous()


def _apply_rotary_torch_bshd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[: x.shape[1]].to(device=x.device, dtype=x.dtype)
    sin = sin[: x.shape[1]].to(device=x.device, dtype=x.dtype)
    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    half = rotary_dim // 2
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    x0 = x_rot[..., :half]
    x1 = x_rot[..., half:]
    out = torch.cat((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1)
    if x_pass.numel():
        out = torch.cat((out, x_pass), dim=-1)
    return out.contiguous()


class VisionAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, projection_size: int | None = None):
        super().__init__()
        if projection_size is None:
            projection_size = embed_dim
        tp = _tp_size()
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.head_dim = projection_size // num_heads
        self.num_heads = num_heads // tp
        self.qkv = QKVParallelLinear(embed_dim, self.head_dim, num_heads, num_heads, bias=True)
        self.proj = RowParallelLinear(projection_size, embed_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        seq_len, batch_size, _ = x.shape
        if max_seqlen is None:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        qkv = self.qkv(x.reshape(seq_len * batch_size, -1))
        if batch_size == 1:
            q, k, v = _pack_qkv_rotary(qkv, self.num_heads, self.head_dim, rotary_pos_emb_cos, rotary_pos_emb_sin)
            out = _flash_attn_varlen(q, k, v, cu_seqlens, max_seqlen, self.head_dim ** -0.5)
            return self.proj(out.reshape(seq_len, batch_size, -1))

        qkv = qkv.view(seq_len, batch_size, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3, 4)
        q = qkv[:, :, 0].contiguous()
        k = qkv[:, :, 1].contiguous()
        v = qkv[:, :, 2].contiguous()
        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            q = _apply_rotary_torch_bshd(q, rotary_pos_emb_cos, rotary_pos_emb_sin)
            k = _apply_rotary_torch_bshd(k, rotary_pos_emb_cos, rotary_pos_emb_sin)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_heads, self.head_dim)
        v = v.reshape(-1, self.num_heads, self.head_dim)
        out = _flash_attn_varlen(q, k, v, cu_seqlens, max_seqlen, self.head_dim ** -0.5)
        return self.proj(out.view(seq_len, batch_size, -1))


class VisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
        bias: bool = True,
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.fc2 = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = _apply_activation_inplace(h, self.act_fn)
        return self.fc2(h)


def _apply_activation_inplace(x: torch.Tensor, act_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    name = act_fn.__class__.__name__.lower()
    if isinstance(act_fn, QuickGELU) or "quickgelu" in name:
        if x.is_cuda and x.is_contiguous():
            n = x.numel()
            with torch.cuda.device(x.device.index if x.device.index is not None else torch.cuda.current_device()):
                _quickgelu_inplace_kernel[(triton.cdiv(n, 1024),)](x, n, BLOCK=1024, num_warps=4)
            return x
        return x.mul_(torch.sigmoid(x * 1.702))
    if isinstance(act_fn, nn.SiLU) or name == "silu":
        if x.is_cuda and x.is_contiguous():
            n = x.numel()
            with torch.cuda.device(x.device.index if x.device.index is not None else torch.cuda.current_device()):
                _silu_inplace_kernel[(triton.cdiv(n, 1024),)](x, n, BLOCK=1024, num_warps=4)
            return x
        return F.silu(x, inplace=True)
    return act_fn(x)


class VisionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim, eps=norm_eps)
        self.norm2 = LayerNorm(embed_dim, eps=norm_eps)
        self.attn = VisionAttention(embed_dim, num_heads)
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim, act_fn=act_fn)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
            max_seqlen,
        )
        x = x + self.mlp(self.norm2(x))
        return x
