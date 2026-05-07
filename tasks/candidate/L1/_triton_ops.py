from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


@triton.jit
def _elementwise_kernel(x_ptr, out_ptr, n_elements, OP: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if OP == "relu":
        y = tl.maximum(x, 0.0)
    elif OP == "sigmoid":
        y = 1.0 / (1.0 + tl.exp(-x))
    elif OP == "silu":
        y = x / (1.0 + tl.exp(-x))
    elif OP == "quickgelu":
        y = x / (1.0 + tl.exp(-1.702 * x))
    elif OP == "gelu":
        y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    elif OP == "gelu_tanh":
        z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        tanh_z = 2.0 / (1.0 + tl.exp(-2.0 * z)) - 1.0
        y = 0.5 * x * (1.0 + tanh_z)
    else:
        y = -tl.log(1.0 + tl.exp(-tl.abs(x))) - tl.maximum(-x, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)


def elementwise(x: torch.Tensor, op: str) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    block = 256 if op == "relu" else (4096 if op == "logsigmoid" else 2048)
    _elementwise_kernel[(triton.cdiv(n, block),)](x, out, n, OP=op, BLOCK=block)
    return out


@triton.jit
def _silu_and_mul_kernel(x_ptr, out_ptr, rows, d: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    row = pid
    mask = offs < d
    gate = tl.load(x_ptr + row * (2 * d) + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * (2 * d) + d + offs, mask=mask, other=0.0).to(tl.float32)
    y = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(out_ptr + row * d + offs, y, mask=mask)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    rows = x.numel() // (2 * d)
    out = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
    block = _next_power_of_2(d)
    _silu_and_mul_kernel[(rows,)](x, out, rows, d, BLOCK=block, num_warps=8)
    return out


@triton.jit
def _norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    rows,
    cols: tl.constexpr,
    eps: tl.constexpr,
    OP: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    INPLACE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        r = tl.load(residual_ptr + row * cols + offs, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        tl.store(residual_ptr + row * cols + offs, x, mask=mask)
    if OP == "layer":
        mean = tl.sum(x, axis=0) / cols
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / cols
        y = centered * tl.rsqrt(var + eps)
    else:
        var = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / cols
        y = x * tl.rsqrt(var + eps)
    if HAS_WEIGHT:
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        y = y * w
    if HAS_BIAS:
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = y + b
    dst = x_ptr if INPLACE else out_ptr
    tl.store(dst + row * cols + offs, y, mask=mask)


def norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    op: str,
    residual: torch.Tensor | None = None,
    inplace: bool = False,
) -> torch.Tensor:
    cols = x.shape[-1]
    rows = x.numel() // cols
    out = x if inplace else torch.empty_like(x)
    block = _next_power_of_2(cols)
    _norm_kernel[(rows,)](
        x,
        residual if residual is not None else x,
        weight if weight is not None else x,
        bias if bias is not None else x,
        out,
        rows,
        cols,
        eps,
        OP=op,
        HAS_RESIDUAL=residual is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        INPLACE=inplace,
        BLOCK=block,
        num_warps=1,
    )
    return out


@triton.jit
def _l2_norm_kernel(x_ptr, out_ptr, rows, cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    inv = tl.rsqrt(tl.maximum(ss, eps * eps))
    tl.store(out_ptr + row * cols + offs, x * inv, mask=mask)


def l2_norm_lastdim(x: torch.Tensor, eps: float) -> torch.Tensor:
    cols = x.shape[-1]
    rows = x.numel() // cols
    out = torch.empty_like(x)
    block = _next_power_of_2(cols)
    _l2_norm_kernel[(rows,)](x, out, rows, cols, eps, BLOCK=block, num_warps=8)
    return out


@triton.jit
def _softmax_lastdim_kernel(x_ptr, out_ptr, rows, cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * cols + offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    ex = tl.exp(x)
    y = ex / tl.sum(ex, axis=0)
    tl.store(out_ptr + row * cols + offs, y, mask=mask)


@triton.jit
def _softmax_dim1_kernel(x_ptr, out_ptr, total, channels: tl.constexpr, inner: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    n = pid // inner
    inner_idx = pid - n * inner
    mask = offs < channels
    ptrs = x_ptr + n * channels * inner + offs * inner + inner_idx
    x = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    ex = tl.exp(x)
    y = ex / tl.sum(ex, axis=0)
    tl.store(out_ptr + n * channels * inner + offs * inner + inner_idx, y, mask=mask)


@triton.jit
def _softmax_dim1_tiled_kernel(
    x_ptr,
    out_ptr,
    channels: tl.constexpr,
    inner: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    n = tl.program_id(0)
    block_inner = tl.program_id(1)
    offs_c = tl.arange(0, BLOCK_C)
    offs_n = block_inner * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_c[:, None] < channels) & (offs_n[None, :] < inner)
    ptrs = x_ptr + n * channels * inner + offs_c[:, None] * inner + offs_n[None, :]
    x = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)[None, :]
    ex = tl.exp(x)
    y = ex / tl.sum(ex, axis=0)[None, :]
    tl.store(out_ptr + n * channels * inner + offs_c[:, None] * inner + offs_n[None, :], y, mask=mask)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    out = torch.empty_like(x)
    if dim < 0:
        dim += x.ndim
    if dim == x.ndim - 1:
        cols = x.shape[-1]
        rows = x.numel() // cols
        block = _next_power_of_2(cols)
        _softmax_lastdim_kernel[(rows,)](x, out, rows, cols, BLOCK=block, num_warps=8)
        return out
    if dim == 1:
        channels = x.shape[1]
        inner = math.prod(x.shape[2:])
        batch = x.shape[0]
        block_c = _next_power_of_2(channels)
        block_n = 32
        _softmax_dim1_tiled_kernel[(batch, triton.cdiv(inner, block_n))](
            x, out, channels, inner, BLOCK_C=block_c, BLOCK_N=block_n, num_warps=4,
        )
        return out
    raise NotImplementedError("Triton softmax candidate supports dim=-1 and dim=1")


@triton.jit
def _topk_softmax_kernel(logits_ptr, weights_ptr, ids_ptr, rows, experts: tl.constexpr, top_k: tl.constexpr, RENORM: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < experts
    logits = tl.load(logits_ptr + row * experts + offs, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    exp_vals = tl.exp(logits - row_max)
    denom_all = tl.sum(exp_vals, axis=0)
    vals = logits
    denom_top = 0.0
    top_offsets = tl.arange(0, 8)
    top_w = tl.zeros((8,), dtype=tl.float32)
    top_ids = tl.zeros((8,), dtype=tl.int32)
    for j in tl.static_range(0, 8):
        maxv = tl.max(vals, axis=0)
        eq = (vals == maxv) & mask
        sel = tl.min(tl.where(eq, offs, experts))
        w = tl.exp(maxv - row_max)
        denom_top += w
        top_w = tl.where(top_offsets == j, w, top_w)
        top_ids = tl.where(top_offsets == j, sel, top_ids)
        vals = tl.where(offs == sel, -float("inf"), vals)
    denom = denom_top if RENORM else denom_all
    tl.store(weights_ptr + row * top_k + top_offsets, top_w / denom)
    tl.store(ids_ptr + row * top_k + top_offsets, top_ids)


def topk_softmax(router_logits: torch.Tensor, top_k: int, renormalize: bool):
    if top_k != 8 or router_logits.shape[-1] != 128:
        raise NotImplementedError("Triton TopKSoftmax candidate is specialized for top_k=8, experts=128")
    rows = router_logits.shape[0]
    weights = torch.empty((rows, top_k), dtype=torch.float32, device=router_logits.device)
    ids = torch.empty((rows, top_k), dtype=torch.int32, device=router_logits.device)
    _topk_softmax_kernel[(rows,)](
        router_logits,
        weights,
        ids,
        rows,
        128,
        top_k,
        RENORM=renormalize,
        BLOCK=128,
        num_warps=2,
    )
    return weights, ids


@triton.jit
def _moe_sum_kernel(inp_ptr, out_ptr, rows, cols: tl.constexpr, topk: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    block_n = tl.program_id(1)
    offs = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < cols
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in tl.static_range(0, 8):
        vals = tl.load(inp_ptr + (row * topk + k) * cols + offs, mask=mask, other=0.0).to(tl.float32)
        acc += vals
    tl.store(out_ptr + row * cols + offs, acc, mask=mask)


def moe_sum(input: torch.Tensor, topk: int) -> torch.Tensor:
    rows = input.shape[0] // topk
    cols = input.shape[1]
    out = torch.empty((rows, cols), dtype=input.dtype, device=input.device)
    block_n = 1024
    _moe_sum_kernel[(rows, triton.cdiv(cols, block_n))](input, out, rows, cols, topk, BLOCK_N=block_n, num_warps=8)
    return out


@triton.jit
def _fill_i32_kernel(ptr, n, value, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offs, value, mask=offs < n)


@triton.jit
def _count_experts_kernel(ids_ptr, counts_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    experts = tl.load(ids_ptr + offs, mask=mask, other=0)
    tl.atomic_add(counts_ptr + experts, 1, sem="relaxed", mask=mask)


@triton.jit
def _prefix_align_kernel(counts_ptr, offsets_ptr, nblocks_ptr, total_ptr, num_experts: tl.constexpr, block_size: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < num_experts
    counts = tl.load(counts_ptr + offs, mask=mask, other=0)
    padded = tl.cdiv(counts, block_size) * block_size
    padded = tl.where(mask, padded, 0)
    csum = tl.cumsum(padded, 0)
    offsets = csum - padded
    tl.store(offsets_ptr + offs, offsets, mask=mask)
    tl.store(nblocks_ptr + offs, padded // block_size, mask=mask)
    total = tl.sum(padded, axis=0)
    tl.store(total_ptr, total)


@triton.jit
def _fill_align_expert_ids_kernel(
    expert_ids_ptr,
    offsets_ptr,
    nblocks_ptr,
    max_blocks,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    block_id = tl.program_id(0)
    eoffs = tl.arange(0, BLOCK_E)
    mask = eoffs < num_experts
    start = tl.load(offsets_ptr + eoffs, mask=mask, other=0) // block_size
    nblocks = tl.load(nblocks_ptr + eoffs, mask=mask, other=0)
    active = mask & (block_id >= start) & (block_id < start + nblocks)
    expert = tl.max(tl.where(active, eoffs, 0), axis=0)
    tl.store(expert_ids_ptr + block_id, expert, mask=block_id < max_blocks)


@triton.jit
def _scatter_align_tokens_kernel(ids_ptr, scratch_ptr, offsets_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    experts = tl.load(ids_ptr + offs, mask=mask, other=0)
    local = tl.atomic_add(scratch_ptr + experts, 1, sem="relaxed", mask=mask)
    base = tl.load(offsets_ptr + experts, mask=mask, other=0)
    tl.store(out_ptr + base + local, offs, mask=mask)


def moe_align(topk_ids: torch.Tensor, block_size: int, num_experts: int, naive: bool = False):
    flat = topk_ids.reshape(-1)
    numel = flat.numel()
    if naive:
        num_tokens = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
        _fill_i32_kernel[(1,)](num_tokens, 1, numel * block_size, BLOCK=1)
        return None, flat, num_tokens

    max_padded = numel * block_size if numel < num_experts else numel + num_experts * (block_size - 1)
    max_blocks = triton.cdiv(max_padded, block_size)
    sorted_token_ids = torch.empty((max_padded,), dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty((max_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    counts = torch.empty((num_experts,), dtype=torch.int32, device=topk_ids.device)
    offsets = torch.empty((num_experts,), dtype=torch.int32, device=topk_ids.device)
    scratch = torch.empty((num_experts,), dtype=torch.int32, device=topk_ids.device)
    nblocks = torch.empty((num_experts,), dtype=torch.int32, device=topk_ids.device)

    block = 1024
    _fill_i32_kernel[(triton.cdiv(num_experts, block),)](counts, num_experts, 0, BLOCK=block)
    _count_experts_kernel[(triton.cdiv(numel, block),)](flat, counts, numel, BLOCK=block)
    block_e = _next_power_of_2(num_experts)
    _prefix_align_kernel[(1,)](counts, offsets, nblocks, num_tokens, num_experts, block_size, BLOCK=block_e)
    _fill_i32_kernel[(triton.cdiv(num_experts, block),)](scratch, num_experts, 0, BLOCK=block)
    _fill_i32_kernel[(triton.cdiv(max_padded, block),)](sorted_token_ids, max_padded, numel, BLOCK=block)
    _fill_align_expert_ids_kernel[(max_blocks,)](
        expert_ids,
        offsets,
        nblocks,
        max_blocks,
        num_experts,
        block_size,
        BLOCK_E=block_e,
        num_warps=4,
    )
    _scatter_align_tokens_kernel[(triton.cdiv(numel, block),)](flat, scratch, offsets, sorted_token_ids, numel, BLOCK=block)
    return sorted_token_ids, expert_ids, num_tokens


@triton.jit
def _embedding_kernel(ids_ptr, weight_ptr, out_ptr, n_ids, dim: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_id = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (pid_id < n_ids) & (offs < dim)
    idx = tl.load(ids_ptr + pid_id)
    vals = tl.load(weight_ptr + idx * dim + offs, mask=mask, other=0.0)
    tl.store(out_ptr + pid_id * dim + offs, vals, mask=mask)


def embedding(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dim = weight.shape[1]
    n_ids = input_ids.numel()
    out = torch.empty((*input_ids.shape, dim), dtype=weight.dtype, device=weight.device)
    block_d = min(1024, _next_power_of_2(dim))
    _embedding_kernel[(n_ids, triton.cdiv(dim, block_d))](input_ids, weight, out, n_ids, dim, BLOCK_D=block_d, num_warps=4)
    return out


@triton.jit
def _one_hot_kernel(x_ptr, out_ptr, n, classes: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = n * classes
    mask = offsets < total
    row = offsets // classes
    cls = offsets - row * classes
    x = tl.load(x_ptr + row, mask=mask, other=-1)
    tl.store(out_ptr + offsets, x == cls, mask=mask)


def one_hot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = torch.empty((*x.shape, num_classes), dtype=torch.int64, device=x.device)
    block = 1024
    _one_hot_kernel[(triton.cdiv(x.numel() * num_classes, block),)](x, out, x.numel(), num_classes, BLOCK=block)
    return out


@triton.jit
def _upsample_nearest2d_kernel(x_ptr, out_ptr, total, channels: tl.constexpr, in_h: tl.constexpr, in_w: tl.constexpr, out_h: tl.constexpr, out_w: tl.constexpr, scale_h: tl.constexpr, scale_w: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    ow = offs % out_w
    tmp = offs // out_w
    oh = tmp % out_h
    tmp = tmp // out_h
    c = tmp % channels
    n = tmp // channels
    ih = oh // scale_h
    iw = ow // scale_w
    vals = tl.load(x_ptr + ((n * channels + c) * in_h + ih) * in_w + iw, mask=mask, other=0.0)
    tl.store(out_ptr + offs, vals, mask=mask)


def upsample_nearest2d(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    n, c, h, w = x.shape
    out = torch.empty((n, c, out_h, out_w), dtype=x.dtype, device=x.device)
    sh = out_h // h
    sw = out_w // w
    block = 256
    total = out.numel()
    _upsample_nearest2d_kernel[(triton.cdiv(total, block),)](x, out, total, c, h, w, out_h, out_w, sh, sw, BLOCK=block)
    return out


@triton.jit
def _batch_norm_eval_kernel(x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr, total, channels: tl.constexpr, hw: tl.constexpr, eps: tl.constexpr, HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    c = (offs // hw) % channels
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    y = (x - mean) * tl.rsqrt(var + eps)
    if HAS_WEIGHT:
        y *= tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    if HAS_BIAS:
        y += tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, y, mask=mask)


def batch_norm_eval(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    total = x.numel()
    channels = x.shape[1]
    hw = x.shape[2] * x.shape[3]
    block = 256
    _batch_norm_eval_kernel[(triton.cdiv(total, block),)](
        x,
        mean,
        var,
        weight if weight is not None else x,
        bias if bias is not None else x,
        out,
        total,
        channels,
        hw,
        eps,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        BLOCK=block,
    )
    return out


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        a = tl.load(a_ptr + offs_m[:, None] * K + k_idxs[None, :], mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_n[None, :] * K + k_idxs[:, None], mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K), other=0.0)
        acc += tl.dot(a, b, input_precision=INPUT_PRECISION)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(input_2d: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    m = input_2d.shape[0]
    k = input_2d.shape[1]
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=input_2d.dtype, device=input_2d.device)
    bm = 16 if m <= 16 else (32 if m <= 128 else 64)
    bn = 64 if n <= 256 else 128
    bk = 64
    _matmul_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        input_2d,
        weight,
        bias if bias is not None else input_2d,
        out,
        m,
        n,
        k,
        HAS_BIAS=bias is not None,
        INPUT_PRECISION="ieee",
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=4 if bm * bn <= 4096 else 8,
        num_stages=3,
    )
    return out


@triton.jit
def _conv1x1_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    OC: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_oc = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    n = tmp // OH
    ih = oh * STRIDE_H - PAD_H
    iw = ow * STRIDE_W - PAD_W
    valid_m = (offs_m < M) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
    offs_c = tl.arange(0, BLOCK_C)
    acc = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)
    for c0 in range(0, C, BLOCK_C):
        cidx = c0 + offs_c
        x = tl.load(
            x_ptr + ((n[:, None] * C + cidx[None, :]) * H + ih[:, None]) * W + iw[:, None],
            mask=valid_m[:, None] & (cidx[None, :] < C),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_oc[None, :] * C + cidx[:, None],
            mask=(offs_oc[None, :] < OC) & (cidx[:, None] < C),
            other=0.0,
        )
        acc += tl.dot(x, w, input_precision="ieee")
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_oc, mask=offs_oc < OC, other=0.0).to(tl.float32)
        acc += bias[None, :]
    tl.store(
        out_ptr + ((n[:, None] * OC + offs_oc[None, :]) * OH + oh[:, None]) * OW + ow[:, None],
        acc,
        mask=(offs_m[:, None] < M) & (offs_oc[None, :] < OC),
    )


@triton.jit
def _depthwise_conv1x1_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    total,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    c = tmp % C
    n = tmp // C
    ih = oh * STRIDE_H - PAD_H
    iw = ow * STRIDE_W - PAD_W
    valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
    x = tl.load(x_ptr + ((n * C + c) * H + ih) * W + iw, mask=valid, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + c, mask=mask, other=0.0).to(tl.float32)
    y = x * w
    if HAS_BIAS:
        y += tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _depthwise_conv2d_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    total,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    c = tmp % C
    n = tmp // C
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for r in tl.static_range(0, 32):
        if r < KH:
            ih = oh * STRIDE_H + r - PAD_H
            valid_h = (ih >= 0) & (ih < H)
            for s in tl.static_range(0, 32):
                if s < KW:
                    iw = ow * STRIDE_W + s - PAD_W
                    valid = mask & valid_h & (iw >= 0) & (iw < W)
                    x = tl.load(
                        x_ptr + ((n * C + c) * H + ih) * W + iw,
                        mask=valid,
                        other=0.0,
                    ).to(tl.float32)
                    wv = tl.load(
                        w_ptr + (c * KH + r) * KW + s,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                    acc += x * wv
    if HAS_BIAS:
        acc += tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, acc, mask=mask)


@triton.jit
def _conv2d_group1_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    OC: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_oc = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    n = tmp // OH
    k_total: tl.constexpr = C * KH * KW
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)
    for k0 in range(0, k_total, BLOCK_K):
        kidx = k0 + offs_k
        kk = kidx % KW
        tmp_k = kidx // KW
        kr = tmp_k % KH
        kc = tmp_k // KH
        ih = oh[:, None] * STRIDE_H + kr[None, :] - PAD_H
        iw = ow[:, None] * STRIDE_W + kk[None, :] - PAD_W
        x = tl.load(
            x_ptr + ((n[:, None] * C + kc[None, :]) * H + ih) * W + iw,
            mask=(
                (offs_m[:, None] < M)
                & (kidx[None, :] < k_total)
                & (ih >= 0)
                & (ih < H)
                & (iw >= 0)
                & (iw < W)
            ),
            other=0.0,
        )
        wv = tl.load(
            w_ptr + offs_oc[None, :] * k_total + kidx[:, None],
            mask=(offs_oc[None, :] < OC) & (kidx[:, None] < k_total),
            other=0.0,
        )
        acc += tl.dot(x, wv, input_precision="ieee")
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_oc, mask=offs_oc < OC, other=0.0).to(tl.float32)
        acc += bias[None, :]
    tl.store(
        out_ptr + ((n[:, None] * OC + offs_oc[None, :]) * OH + oh[:, None]) * OW + ow[:, None],
        acc,
        mask=(offs_m[:, None] < M) & (offs_oc[None, :] < OC),
    )


@triton.jit
def _conv2d_group1_scalar_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    total,
    C: tl.constexpr,
    OC: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    oc = tmp % OC
    n = tmp // OC
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ci in range(0, C):
        for r in range(0, KH):
            ih = oh * STRIDE_H + r - PAD_H
            valid_h = (ih >= 0) & (ih < H)
            for s in range(0, KW):
                iw = ow * STRIDE_W + s - PAD_W
                valid = mask & valid_h & (iw >= 0) & (iw < W)
                xv = tl.load(
                    x_ptr + ((n * C + ci) * H + ih) * W + iw,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                wv = tl.load(
                    w_ptr + ((oc * C + ci) * KH + r) * KW + s,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                acc += xv * wv
    if HAS_BIAS:
        acc += tl.load(bias_ptr + oc, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, acc, mask=mask)


@triton.jit
def _conv1x1_smallc_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    total,
    C: tl.constexpr,
    OC: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    oc = tmp % OC
    n = tmp // OC
    ih = oh * STRIDE_H - PAD_H
    iw = ow * STRIDE_W - PAD_W
    valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ci in tl.static_range(0, 16):
        if ci < C:
            xv = tl.load(x_ptr + ((n * C + ci) * H + ih) * W + iw, mask=valid, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + oc * C + ci, mask=mask, other=0.0).to(tl.float32)
            xv = tl.inline_asm_elementwise(
                "cvt.rna.tf32.f32 $0, $1;",
                constraints="=f,f",
                args=[xv],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            wv = tl.inline_asm_elementwise(
                "cvt.rna.tf32.f32 $0, $1;",
                constraints="=f,f",
                args=[wv],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            acc += xv * wv
    if HAS_BIAS:
        acc += tl.load(bias_ptr + oc, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, acc, mask=mask)


def conv2d_1x1(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    groups: int,
) -> torch.Tensor:
    n, c, h, w = x.shape
    oc = weight.shape[0]
    kh, kw = weight.shape[-2:]
    sh, sw = int(stride[0]), int(stride[1])
    ph, pw = int(padding[0]), int(padding[1])
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    out = torch.empty((n, oc, oh, ow), dtype=x.dtype, device=x.device)
    if kh != 1 or kw != 1:
        if groups == c and oc == c:
            block = 256
            _depthwise_conv2d_kernel[(triton.cdiv(out.numel(), block),)](
                x,
                weight,
                bias if bias is not None else x,
                out,
                out.numel(),
                c,
                h,
                w,
                oh,
                ow,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                HAS_BIAS=bias is not None,
                BLOCK=block,
            )
            return out
        if groups != 1:
            raise NotImplementedError("Triton candidate conv2d supports groups=1 or depthwise")
        if x.dtype == torch.float32 and c <= 16 and kh <= 32 and kw <= 32:
            block = 128
            _conv2d_group1_scalar_kernel[(triton.cdiv(out.numel(), block),)](
                x,
                weight,
                bias if bias is not None else x,
                out,
                out.numel(),
                c,
                oc,
                h,
                w,
                oh,
                ow,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                HAS_BIAS=bias is not None,
                BLOCK=block,
                num_warps=4,
            )
            return out
        m = n * oh * ow
        bm = 16
        boc = 32 if oc <= 256 else 16
        bk = 64
        _conv2d_group1_kernel[(triton.cdiv(m, bm), triton.cdiv(oc, boc))](
            x,
            weight,
            bias if bias is not None else x,
            out,
            m,
            c,
            oc,
            h,
            w,
            oh,
            ow,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            HAS_BIAS=bias is not None,
            BLOCK_M=bm,
            BLOCK_OC=boc,
            BLOCK_K=bk,
            num_warps=4,
            num_stages=3,
        )
        return out
    if groups == c and oc == c:
        total = out.numel()
        block = 256
        _depthwise_conv1x1_kernel[(triton.cdiv(total, block),)](
            x,
            weight,
            bias if bias is not None else x,
            out,
            total,
            c,
            h,
            w,
            oh,
            ow,
            sh,
            sw,
            ph,
            pw,
            HAS_BIAS=bias is not None,
            BLOCK=block,
        )
        return out
    if groups != 1:
        raise NotImplementedError("Triton candidate conv2d supports groups=1 or depthwise")
    if c < 16:
        block = 256
        _conv1x1_smallc_kernel[(triton.cdiv(out.numel(), block),)](
            x,
            weight,
            bias if bias is not None else x,
            out,
            out.numel(),
            c,
            oc,
            h,
            w,
            oh,
            ow,
            sh,
            sw,
            ph,
            pw,
            HAS_BIAS=bias is not None,
            BLOCK=block,
        )
        return out
    m = n * oh * ow
    bm = 16
    boc = 32 if oc <= 128 else 16
    bc = max(16, min(256, _next_power_of_2(c)))
    _conv1x1_kernel[(triton.cdiv(m, bm), triton.cdiv(oc, boc))](
        x,
        weight,
        bias if bias is not None else x,
        out,
        m,
        c,
        oc,
        h,
        w,
        oh,
        ow,
        sh,
        sw,
        ph,
        pw,
        HAS_BIAS=bias is not None,
        BLOCK_M=bm,
        BLOCK_OC=boc,
        BLOCK_C=bc,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def _max_pool2d_kernel(
    x_ptr,
    out_ptr,
    total,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    c = tmp % C
    n = tmp // C
    acc = tl.full((BLOCK,), -float("inf"), dtype=tl.float32)
    for r in tl.static_range(0, 5):
        for s in tl.static_range(0, 5):
            ih = oh * SH + r - PH
            iw = ow * SW + s - PW
            valid = mask & (r < KH) & (s < KW) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            vals = tl.load(x_ptr + ((n * C + c) * H + ih) * W + iw, mask=valid, other=-float("inf")).to(tl.float32)
            acc = tl.maximum(acc, vals)
    tl.store(out_ptr + offs, acc, mask=mask)


def max_pool2d(x: torch.Tensor, kernel_size: tuple[int, int], stride: tuple[int, int], padding: tuple[int, int], ceil_mode: bool) -> torch.Tensor:
    if ceil_mode:
        raise NotImplementedError("Triton candidate max_pool2d only supports ceil_mode=False")
    n, c, h, w = x.shape
    kh, kw = int(kernel_size[0]), int(kernel_size[1])
    sh, sw = int(stride[0]), int(stride[1])
    ph, pw = int(padding[0]), int(padding[1])
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    out = torch.empty((n, c, oh, ow), dtype=x.dtype, device=x.device)
    block = 256
    _max_pool2d_kernel[(triton.cdiv(out.numel(), block),)](
        x,
        out,
        out.numel(),
        c,
        h,
        w,
        oh,
        ow,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        BLOCK=block,
    )
    return out


@triton.jit
def _dense_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    B: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    q_block = tl.program_id(2)
    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    q = tl.load(
        q_ptr + ((b * L + offs_m[:, None]) * H + h) * D + offs_d[None, :],
        mask=(offs_m[:, None] < L) & mask_d[None, :],
        other=0.0,
    )
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, L, BLOCK_N):
        k_idx = start_n + offs_n
        k = tl.load(
            k_ptr + ((b * L + k_idx[:, None]) * H + h) * D + offs_d[None, :],
            mask=(k_idx[:, None] < L) & mask_d[None, :],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        valid = (offs_m[:, None] < L) & (k_idx[None, :] < L)
        if CAUSAL:
            valid = valid & (k_idx[None, :] <= offs_m[:, None])
        scores = tl.where(valid, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        v = tl.load(
            v_ptr + ((b * L + k_idx[:, None]) * H + h) * D + offs_d[None, :],
            mask=(k_idx[:, None] < L) & mask_d[None, :],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(
        out_ptr + ((b * L + offs_m[:, None]) * H + h) * D + offs_d[None, :],
        out,
        mask=(offs_m[:, None] < L) & mask_d[None, :],
    )


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float | None, causal: bool) -> torch.Tensor:
    if q.shape != k.shape or q.shape != v.shape:
        raise NotImplementedError("dense attention Triton path requires matching Q/K/V shapes")
    b, l, h, d = q.shape
    out = torch.empty_like(q)
    scale = softmax_scale if softmax_scale is not None else d ** -0.5
    block_d = _next_power_of_2(d)
    block_m = 128
    block_n = 64
    _dense_attention_kernel[(b, h, triton.cdiv(l, block_m))](
        q,
        k,
        v,
        out,
        b,
        l,
        h,
        d,
        float(scale),
        CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def _varlen_attention_small_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cu_q_ptr,
    cu_k_ptr,
    out_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    MAX_Q: tl.constexpr,
    MAX_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    qpos = tl.program_id(2)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)
    mask_d = offs_d < D
    q_start = tl.load(cu_q_ptr + b)
    q_end = tl.load(cu_q_ptr + b + 1)
    k_start = tl.load(cu_k_ptr + b)
    k_end = tl.load(cu_k_ptr + b + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    active_q = qpos < q_len
    q_abs = q_start + qpos
    q = tl.load(q_ptr + (q_abs * H + h) * D + offs_d, mask=active_q & mask_d, other=0.0).to(tl.float32)
    k_abs = k_start + offs_k
    k = tl.load(
        k_ptr + (k_abs[:, None] * H + h) * D + offs_d[None, :],
        mask=(offs_k[:, None] < k_len) & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(k * q[None, :], axis=1) * SCALE
    valid_k = offs_k < k_len
    if CAUSAL:
        q_aligned = qpos + (k_len - q_len)
        valid_k = valid_k & (offs_k <= q_aligned)
    scores = tl.where(active_q & valid_k, scores, -float("inf"))
    scores = scores - tl.max(scores, axis=0)
    probs = tl.exp(scores)
    probs = probs / tl.sum(probs, axis=0)
    v = tl.load(
        v_ptr + (k_abs[:, None] * H + h) * D + offs_d[None, :],
        mask=(offs_k[:, None] < k_len) & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(probs[:, None] * v, axis=0)
    tl.store(out_ptr + (q_abs * H + h) * D + offs_d, acc, mask=active_q & mask_d)


def varlen_attention_small(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise NotImplementedError("small varlen attention requires matching Q/K/V heads")
    if max_seqlen_q > 64 or max_seqlen_k > 64:
        raise NotImplementedError("small varlen attention is specialized for sequence length <= 64")
    out = torch.empty_like(q)
    b = cu_seqlens_q.numel() - 1
    h = q.shape[1]
    d = q.shape[2]
    block_k = _next_power_of_2(max_seqlen_k)
    block_d = _next_power_of_2(d)
    _varlen_attention_small_kernel[(b, h, max_seqlen_q)](
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        out,
        b,
        h,
        d,
        float(softmax_scale),
        CAUSAL=causal,
        MAX_Q=max_seqlen_q,
        MAX_K=max_seqlen_k,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


@triton.jit
def _varlen_attention_block_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cu_q_ptr,
    cu_k_ptr,
    out_ptr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    MAX_Q: tl.constexpr,
    MAX_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    kvh = qh // (QH // KVH)
    q_block = tl.program_id(2)
    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    q_start = tl.load(cu_q_ptr + b)
    q_end = tl.load(cu_q_ptr + b + 1)
    k_start = tl.load(cu_k_ptr + b)
    k_end = tl.load(cu_k_ptr + b + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    q_abs = q_start + offs_m
    q = tl.load(
        q_ptr + (q_abs[:, None] * QH + qh) * D + offs_d[None, :],
        mask=(offs_m[:, None] < q_len) & mask_d[None, :],
        other=0.0,
    )
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for start_n in range(0, MAX_K, BLOCK_N):
        k_idx = start_n + offs_n
        k_abs = k_start + k_idx
        k = tl.load(
            k_ptr + (k_abs[:, None] * KVH + kvh) * D + offs_d[None, :],
            mask=(k_idx[:, None] < k_len) & mask_d[None, :],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        valid = (offs_m[:, None] < q_len) & (k_idx[None, :] < k_len)
        if CAUSAL:
            q_aligned = offs_m + (k_len - q_len)
            valid = valid & (k_idx[None, :] <= q_aligned[:, None])
            if WINDOW_LEFT >= 0:
                valid = valid & (k_idx[None, :] >= (q_aligned[:, None] - WINDOW_LEFT))
        scores = tl.where(valid, scores, -1.0e20)
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(valid, p, 0.0)
        safe_m_i = tl.where(m_i != -float("inf"), m_i, m_new - 80.0)
        alpha = tl.exp(safe_m_i - m_new)
        v = tl.load(
            v_ptr + (k_abs[:, None] * KVH + kvh) * D + offs_d[None, :],
            mask=(k_idx[:, None] < k_len) & mask_d[None, :],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(
        out_ptr + (q_abs[:, None] * QH + qh) * D + offs_d[None, :],
        out,
        mask=(offs_m[:, None] < q_len) & mask_d[None, :],
    )


def varlen_attention_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_left: int = -1,
) -> torch.Tensor:
    if k.shape[1] != v.shape[1] or q.shape[1] % k.shape[1] != 0:
        raise NotImplementedError("block varlen attention requires Q heads to be a multiple of KV heads")
    out = torch.empty_like(q)
    b = cu_seqlens_q.numel() - 1
    qh = q.shape[1]
    kvh = k.shape[1]
    d = q.shape[2]
    block_d = _next_power_of_2(d)
    block_m = 16 if max_seqlen_q <= 64 else 64
    block_n = 64
    _varlen_attention_block_kernel[(b, qh, triton.cdiv(max_seqlen_q, block_m))](
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        out,
        qh,
        kvh,
        d,
        float(softmax_scale),
        CAUSAL=causal,
        WINDOW_LEFT=int(window_left),
        MAX_Q=max_seqlen_q,
        MAX_K=max_seqlen_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def _paged_varlen_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cu_q_ptr,
    cu_k_ptr,
    block_table_ptr,
    out_ptr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    MAX_Q: tl.constexpr,
    MAX_K: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    HND: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    q_block = tl.program_id(2)
    kvh = qh // (QH // KVH)
    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    q_start = tl.load(cu_q_ptr + b)
    q_end = tl.load(cu_q_ptr + b + 1)
    k_start = tl.load(cu_k_ptr + b)
    k_end = tl.load(cu_k_ptr + b + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    q_abs = q_start + offs_m
    q = tl.load(
        q_ptr + (q_abs[:, None] * QH + qh) * D + offs_d[None, :],
        mask=(offs_m[:, None] < q_len) & mask_d[None, :],
        other=0.0,
    )
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, MAX_K, BLOCK_N):
        k_idx = start_n + offs_n
        page_idx = k_idx // PAGE
        slots = k_idx - page_idx * PAGE
        valid_k = k_idx < k_len
        block_ids = tl.load(
            block_table_ptr + b * tl.cdiv(MAX_K, PAGE) + page_idx,
            mask=valid_k,
            other=0,
        )
        if HND:
            k_ptrs = k_ptr + (((block_ids[:, None] * KVH + kvh) * PAGE + slots[:, None]) * D + offs_d[None, :])
            v_ptrs = v_ptr + (((block_ids[:, None] * KVH + kvh) * PAGE + slots[:, None]) * D + offs_d[None, :])
        else:
            k_ptrs = k_ptr + (((block_ids[:, None] * PAGE + slots[:, None]) * KVH + kvh) * D + offs_d[None, :])
            v_ptrs = v_ptr + (((block_ids[:, None] * PAGE + slots[:, None]) * KVH + kvh) * D + offs_d[None, :])
        k = tl.load(k_ptrs, mask=valid_k[:, None] & mask_d[None, :], other=0.0)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        valid = (offs_m[:, None] < q_len) & valid_k[None, :]
        if CAUSAL:
            q_aligned = offs_m + (k_len - q_len)
            valid = valid & (k_idx[None, :] <= q_aligned[:, None])
        scores = tl.where(valid, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        v = tl.load(v_ptrs, mask=valid_k[:, None] & mask_d[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(
        out_ptr + (q_abs[:, None] * QH + qh) * D + offs_d[None, :],
        out,
        mask=(offs_m[:, None] < q_len) & mask_d[None, :],
    )


def paged_varlen_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    hnd: bool,
) -> torch.Tensor:
    out = torch.empty_like(q)
    qh = q.shape[1]
    kvh = k_cache.shape[1] if hnd else k_cache.shape[-2]
    d = q.shape[2]
    block_d = _next_power_of_2(d)
    block_m = 32 if d >= 128 else 64
    block_n = 64
    batch = cu_seqlens_q.numel() - 1
    _paged_varlen_attention_kernel[(batch, qh, triton.cdiv(max_seqlen_q, block_m))](
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        block_table,
        out,
        qh,
        kvh,
        d,
        int(max_seqlen_q),
        int(max_seqlen_k),
        float(softmax_scale),
        CAUSAL=causal,
        HND=hnd,
        PAGE=16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def _rope_inplace_kernel(
    x_ptr,
    pos_ptr,
    cache_ptr,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    half: tl.constexpr,
    IS_NEOX: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tokens * heads * half
    mask = offs < total
    dim = offs % half
    tmp = offs // half
    head = tmp % heads
    token = tmp // heads
    pos = tl.load(pos_ptr + token, mask=mask, other=0)
    cos = tl.load(cache_ptr + pos * head_dim + dim, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(cache_ptr + pos * head_dim + half + dim, mask=mask, other=0.0).to(tl.float32)
    if IS_NEOX:
        d0 = dim
        d1 = dim + half
    else:
        d0 = dim * 2
        d1 = dim * 2 + 1
    base = (token * heads + head) * head_dim
    x0 = tl.load(x_ptr + base + d0, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base + d1, mask=mask, other=0.0).to(tl.float32)
    p0 = x0 * cos
    p1 = x1 * sin
    p2 = x1 * cos
    p3 = x0 * sin
    if IS_BF16:
        p0_bits = p0.to(tl.uint32, bitcast=True)
        p1_bits = p1.to(tl.uint32, bitcast=True)
        p2_bits = p2.to(tl.uint32, bitcast=True)
        p3_bits = p3.to(tl.uint32, bitcast=True)
        p0 = ((p0_bits + 32767 + ((p0_bits >> 16) & 1)) & 4294901760).to(tl.float32, bitcast=True)
        p1 = ((p1_bits + 32767 + ((p1_bits >> 16) & 1)) & 4294901760).to(tl.float32, bitcast=True)
        p2 = ((p2_bits + 32767 + ((p2_bits >> 16) & 1)) & 4294901760).to(tl.float32, bitcast=True)
        p3 = ((p3_bits + 32767 + ((p3_bits >> 16) & 1)) & 4294901760).to(tl.float32, bitcast=True)
    out0 = p0 - p1
    out1 = p2 + p3
    if IS_BF16:
        out0_bits = out0.to(tl.uint32, bitcast=True)
        out1_bits = out1.to(tl.uint32, bitcast=True)
        out0 = ((out0_bits + 32767 + ((out0_bits >> 16) & 1)) & 4294901760).to(tl.float32, bitcast=True)
        out1 = ((out1_bits + 32767 + ((out1_bits >> 16) & 1)) & 4294901760).to(tl.float32, bitcast=True)
    tl.store(x_ptr + base + d0, out0, mask=mask)
    tl.store(x_ptr + base + d1, out1, mask=mask)


def apply_rope_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_dim: int,
    cache: torch.Tensor,
    is_neox_style: bool,
    round_bf16: bool | None = None,
):
    if round_bf16 is None:
        round_bf16 = query.dtype is torch.bfloat16
    tokens = positions.numel()
    half = head_dim // 2
    block = 256
    q_heads = query.numel() // (tokens * head_dim)
    _rope_inplace_kernel[(triton.cdiv(tokens * q_heads * half, block),)](
        query,
        positions,
        cache,
        tokens,
        q_heads,
        head_dim,
        half,
        IS_NEOX=is_neox_style,
        IS_BF16=round_bf16,
        BLOCK=block,
    )
    if key is not None:
        k_heads = key.numel() // (tokens * head_dim)
        _rope_inplace_kernel[(triton.cdiv(tokens * k_heads * half, block),)](
            key,
            positions,
            cache,
            tokens,
            k_heads,
            head_dim,
            half,
            IS_NEOX=is_neox_style,
            IS_BF16=round_bf16,
            BLOCK=block,
        )
    return query, key


@triton.jit
def _mrope_interleaved_inplace_kernel(
    x_ptr,
    pos_ptr,
    cache_ptr,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    half: tl.constexpr,
    section1: tl.constexpr,
    section2: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tokens * heads * half
    mask = offs < total
    dim = offs % half
    tmp = offs // half
    head = tmp % heads
    token = tmp // heads
    axis = tl.full((BLOCK,), 0, dtype=tl.int64)
    axis = tl.where((dim < section1 * 3) & ((dim % 3) == 1), 1, axis)
    axis = tl.where((dim < section2 * 3) & ((dim % 3) == 2), 2, axis)
    pos = tl.load(pos_ptr + axis * tokens + token, mask=mask, other=0)
    cos = tl.load(cache_ptr + pos * head_dim + dim, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(cache_ptr + pos * head_dim + half + dim, mask=mask, other=0.0).to(tl.float32)
    base = (token * heads + head) * head_dim
    x0 = tl.load(x_ptr + base + dim, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base + half + dim, mask=mask, other=0.0).to(tl.float32)
    p0 = x0 * cos
    p1 = x1 * sin
    p2 = x1 * cos
    p3 = x0 * sin
    if IS_BF16:
        p0_bits = p0.to(tl.uint32, bitcast=True)
        p1_bits = p1.to(tl.uint32, bitcast=True)
        p2_bits = p2.to(tl.uint32, bitcast=True)
        p3_bits = p3.to(tl.uint32, bitcast=True)
        p0 = ((p0_bits + 32767 + ((p0_bits >> 16) & 1)) & 4294901760).to(tl.float32, bitcast=True)
        p1 = ((p1_bits + 32767 + ((p1_bits >> 16) & 1)) & 4294901760).to(tl.float32, bitcast=True)
        p2 = ((p2_bits + 32767 + ((p2_bits >> 16) & 1)) & 4294901760).to(tl.float32, bitcast=True)
        p3 = ((p3_bits + 32767 + ((p3_bits >> 16) & 1)) & 4294901760).to(tl.float32, bitcast=True)
    out0 = p0 - p1
    out1 = p2 + p3
    if IS_BF16:
        out0_bits = out0.to(tl.uint32, bitcast=True)
        out1_bits = out1.to(tl.uint32, bitcast=True)
        out0 = ((out0_bits + 32767 + ((out0_bits >> 16) & 1)) & 4294901760).to(tl.float32, bitcast=True)
        out1 = ((out1_bits + 32767 + ((out1_bits >> 16) & 1)) & 4294901760).to(tl.float32, bitcast=True)
    tl.store(x_ptr + base + dim, out0, mask=mask)
    tl.store(x_ptr + base + half + dim, out1, mask=mask)


def apply_mrope_interleaved_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    cache: torch.Tensor,
    section: list[int] | tuple[int, ...],
):
    tokens = positions.shape[-1]
    half = head_dim // 2
    block = 256
    q_heads = query.numel() // (tokens * head_dim)
    _mrope_interleaved_inplace_kernel[(triton.cdiv(tokens * q_heads * half, block),)](
        query,
        positions,
        cache,
        tokens,
        q_heads,
        head_dim,
        half,
        int(section[1]),
        int(section[2]),
        IS_BF16=query.dtype is torch.bfloat16,
        BLOCK=block,
    )
    k_heads = key.numel() // (tokens * head_dim)
    _mrope_interleaved_inplace_kernel[(triton.cdiv(tokens * k_heads * half, block),)](
        key,
        positions,
        cache,
        tokens,
        k_heads,
        head_dim,
        half,
        int(section[1]),
        int(section[2]),
        IS_BF16=key.dtype is torch.bfloat16,
        BLOCK=block,
    )
    return query, key


@triton.jit
def _decode_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    seqlens_ptr,
    block_table_ptr,
    out_ptr,
    B: tl.constexpr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    MAX_SEQ: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_BLOCK_TABLE: tl.constexpr,
    HND: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    kvh = qh // (QH // KVH)
    q = tl.load(q_ptr + (b * QH + qh) * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    seq_len = tl.load(seqlens_ptr + b)
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for start in range(0, MAX_SEQ, BLOCK_M):
        offs_m = start + tl.arange(0, BLOCK_M)
        valid_m = offs_m < seq_len
        if HAS_BLOCK_TABLE:
            block_ids = tl.load(block_table_ptr + b * tl.cdiv(MAX_SEQ, PAGE) + (offs_m // PAGE), mask=valid_m, other=0)
            slots = offs_m - (offs_m // PAGE) * PAGE
            if HND:
                ptrs = k_ptr + (((block_ids[:, None] * KVH + kvh) * PAGE + slots[:, None]) * D + offs_d[None, :])
                vptrs = v_ptr + (((block_ids[:, None] * KVH + kvh) * PAGE + slots[:, None]) * D + offs_d[None, :])
            else:
                ptrs = k_ptr + (((block_ids[:, None] * PAGE + slots[:, None]) * KVH + kvh) * D + offs_d[None, :])
                vptrs = v_ptr + (((block_ids[:, None] * PAGE + slots[:, None]) * KVH + kvh) * D + offs_d[None, :])
        else:
            ptrs = k_ptr + (((b * MAX_SEQ + offs_m[:, None]) * KVH + kvh) * D + offs_d[None, :])
            vptrs = v_ptr + (((b * MAX_SEQ + offs_m[:, None]) * KVH + kvh) * D + offs_d[None, :])
        k = tl.load(ptrs, mask=valid_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * SCALE
        scores = tl.where(valid_m, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)
        v = tl.load(vptrs, mask=valid_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
    out = acc / l_i
    tl.store(out_ptr + (b * QH + qh) * D + offs_d, out, mask=mask_d)


@triton.jit
def _decode_attention_grouped_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    seqlens_ptr,
    block_table_ptr,
    out_ptr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    MAX_SEQ: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_BLOCK_TABLE: tl.constexpr,
    HND: tl.constexpr,
    PAGE: tl.constexpr,
    GROUP_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    offs_g = tl.arange(0, GROUP_Q)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    qh = kvh * GROUP_Q + offs_g
    q = tl.load(
        q_ptr + (b * QH + qh[:, None]) * D + offs_d[None, :],
        mask=mask_d[None, :],
        other=0.0,
    )
    seq_len = tl.load(seqlens_ptr + b)
    m_i = tl.full((GROUP_Q,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((GROUP_Q,), dtype=tl.float32)
    acc = tl.zeros((GROUP_Q, BLOCK_D), dtype=tl.float32)

    for start in range(0, MAX_SEQ, BLOCK_M):
        offs_m = start + tl.arange(0, BLOCK_M)
        valid_m = offs_m < seq_len
        if HAS_BLOCK_TABLE:
            page_idx = offs_m // PAGE
            block_ids = tl.load(
                block_table_ptr + b * tl.cdiv(MAX_SEQ, PAGE) + page_idx,
                mask=valid_m,
                other=0,
            )
            slots = offs_m - page_idx * PAGE
            if HND:
                ptrs = k_ptr + (((block_ids[:, None] * KVH + kvh) * PAGE + slots[:, None]) * D + offs_d[None, :])
                vptrs = v_ptr + (((block_ids[:, None] * KVH + kvh) * PAGE + slots[:, None]) * D + offs_d[None, :])
            else:
                ptrs = k_ptr + (((block_ids[:, None] * PAGE + slots[:, None]) * KVH + kvh) * D + offs_d[None, :])
                vptrs = v_ptr + (((block_ids[:, None] * PAGE + slots[:, None]) * KVH + kvh) * D + offs_d[None, :])
        else:
            ptrs = k_ptr + (((b * MAX_SEQ + offs_m[:, None]) * KVH + kvh) * D + offs_d[None, :])
            vptrs = v_ptr + (((b * MAX_SEQ + offs_m[:, None]) * KVH + kvh) * D + offs_d[None, :])
        k = tl.load(ptrs, mask=valid_m[:, None] & mask_d[None, :], other=0.0)
        scores = tl.dot(k, tl.trans(q), input_precision="ieee") * SCALE
        scores = tl.where(valid_m[:, None], scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new[None, :])
        alpha = tl.exp(m_i - m_new)
        v = tl.load(vptrs, mask=valid_m[:, None] & mask_d[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(tl.trans(p.to(tl.bfloat16)), v, input_precision="ieee")
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(
        out_ptr + (b * QH + qh[:, None]) * D + offs_d[None, :],
        out,
        mask=mask_d[None, :],
    )


def decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, cache_seqlens: torch.Tensor, softmax_scale: float, block_table: torch.Tensor | None = None, hnd: bool = False) -> torch.Tensor:
    b, qh, d = q.shape
    kvh = k_cache.shape[1] if hnd else k_cache.shape[-2]
    if block_table is None:
        max_seq = k_cache.shape[2] if hnd else k_cache.shape[1]
    else:
        max_seq = block_table.shape[1] * 16
    out = torch.empty_like(q)
    block_d = _next_power_of_2(d)
    group_q = qh // kvh if qh % kvh == 0 else 1
    block_m = 256
    if group_q > 1 and group_q <= 8:
        _decode_attention_grouped_kernel[(b, kvh)](
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            block_table if block_table is not None else q,
            out,
            qh,
            kvh,
            d,
            max_seq,
            float(softmax_scale),
            HAS_BLOCK_TABLE=block_table is not None,
            HND=hnd,
            PAGE=16,
            GROUP_Q=group_q,
            BLOCK_M=block_m if d <= 64 else 64,
            BLOCK_D=block_d,
            num_warps=4,
        )
        return out
    _decode_attention_kernel[(b, qh)](
        q,
        k_cache,
        v_cache,
        cache_seqlens,
        block_table if block_table is not None else q,
        out,
        b,
        qh,
        kvh,
        d,
        max_seq,
        float(softmax_scale),
        HAS_BLOCK_TABLE=block_table is not None,
        HND=hnd,
        PAGE=16,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out
