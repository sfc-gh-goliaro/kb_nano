import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _block_inclusive_scan_rows(
    x_ptr,              # *fp32, shape [M, N]
    y_ptr,              # *fp32, shape [M, N] (temporary: block results)
    block_sums_ptr,     # *fp32, shape [M, B]
    M, N,               # ints
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    # Number of blocks along N
    num_blocks = (N + BLOCK_N - 1) // BLOCK_N

    for blk in range(0, num_blocks):
        start = blk * BLOCK_N
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N

        # Load values for this block (out-of-bounds => 0)
        ptrs = x_ptr + row * stride_xm + offs * stride_xn
        v = tl.load(ptrs, mask=mask, other=0.0)

        # Sequential inclusive scan within the block
        running = 0.0
        out = tl.zeros([BLOCK_N], dtype=v.dtype)
        for j in range(0, BLOCK_N):
            inc = v[j]
            running = running + inc
            out[j] = running

        # Store out to y
        y_ptrs = y_ptr + row * stride_ym + offs * stride_yn
        tl.store(y_ptrs, out, mask=mask)

        # Block sum = last element of out (valid due to mask and zero-padding)
        block_sum = out[BLOCK_N - 1]
        # Store block sum at [row, blk]
        tl.store(block_sums_ptr + row * num_blocks + blk, block_sum)


@triton.jit
def _blocksum_exclusive_scan(
    block_sums_ptr,     # *fp32, shape [M*B]
    block_prefix_ptr,   # *fp32, shape [M*B] (will store p; we'll write p+s afterwards)
    M, B,               # ints
    BLOCK_B: tl.constexpr,
):
    row = tl.program_id(0)

    # View of this row's block sums: index base = row*B
    base = row * B

    # Working vectors
    idx = tl.arange(0, BLOCK_B)
    mask = idx < B

    p = tl.zeros([BLOCK_B], dtype=tl.float32)
    s = tl.load(block_sums_ptr + base + idx, mask=mask, other=0.0)

    step = 1
    while step < B:
        # prev = p before update
        prev = p
        # shifted = p delayed by 'step' where idx >= step, else 0
        shifted = tl.where(idx >= step, p, 0.0)
        # p = p + prev shifted right by 'step'
        # Emulate shift: p' = p; p = p + prev shifted
        # But we need to add prev at positions < step: add prev where idx < step
        add_mask = idx < step
        add_val = tl.where(add_mask, prev, 0.0)
        p = p + add_val
        step = step * 2

    # Inclusive block prefixes: p + s
    incl = p + s
    # Store to block_prefix_ptr[row*B : row*B+B]
    tl.store(block_prefix_ptr + base + idx, incl, mask=mask)


@triton.jit
def _add_block_offsets(
    y_ptr,              # *fp32, shape [M, N]
    block_prefix_ptr,   # *fp32, shape [M, B]
    M, N,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    num_blocks = (N + BLOCK_N - 1) // BLOCK_N

    for blk in range(0, num_blocks):
        start = blk * BLOCK_N
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N

        # Load block offset p[blk]
        offset = tl.load(block_prefix_ptr + row * num_blocks + blk)
        # Load y block
        y_ptrs = y_ptr + row * stride_ym + offs * stride_yn
        y_block = tl.load(y_ptrs, mask=mask, other=0.0)
        # Add offset
        y_block = y_block + offset
        # Store back
        tl.store(y_ptrs, y_block, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        # Tunables
        self.block_n = 2048
        self.num_warps = 4
        self.num_stages = 2

    def forward(self, x: torch.Tensor):
        # Fallbacks and checks
        if not TRITON_AVAILABLE:
            return torch.cumsum(x, dim=self.dim)
        if not x.is_cuda:
            return torch.cumsum(x, dim=self.dim)
        if x.dtype != torch.float32:
            # Keep it simple: use float32
            x = x.to(torch.float32)

        # Only support 2D and dim=1 for this kernel
        if x.dim() != 2 or self.dim != 1:
            return torch.cumsum(x, dim=self.dim)

        # Make contiguous
        if not x.is_contiguous():
            x = x.contiguous()

        M, N = x.shape
        # Allocate temporaries
        device = x.device
        y = torch.empty_like(x)
        # block_sums: [M, B]
        BLOCK_N = self.block_n
        B = (N + BLOCK_N - 1) // BLOCK_N
        block_sums = torch.empty((M, B), dtype=torch.float32, device=device)
        block_prefix = torch.empty((M, B), dtype=torch.float32, device=device)

        # Strides (elements)
        stride_xm, stride_xn = x.stride(0), x.stride(1)
        stride_ym, stride_yn = y.stride(0), y.stride(1)

        # Grid: one program per row
        grid = (M,)

        # Step 1: block inclusive scan into y, and block sums
        _block_inclusive_scan_rows[grid](
            x, y, block_sums,
            M, N,
            stride_xm, stride_xn,
            stride_ym, stride_yn,
            BLOCK_N=BLOCK_N,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # Step 2: exclusive scan of block sums to get block prefixes
        # Flatten block arrays to 1D for simplicity
        block_sums_flat = block_sums.view(-1)
        block_prefix_flat = block_prefix.view(-1)
        BLOCK_B = 1
        # Choose BLOCK_B as a power-of-two >= B, capped
        # But here B is small; using B is fine
        while BLOCK_B < B:
            BLOCK_B *= 2
        _blocksum_exclusive_scan[grid](
            block_sums_flat, block_prefix_flat,
            M, B,
            BLOCK_B=BLOCK_B,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # Step 3: add block offsets to y
        _add_block_offsets[grid](
            y, block_prefix,
            M, N,
            stride_ym, stride_yn,
            BLOCK_N=BLOCK_N,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        return y
