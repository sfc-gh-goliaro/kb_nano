import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _block_inclusive_scan_rows_scalar(
    x_ptr,              # *fp32, shape [M, N]
    y_ptr,              # *fp32, shape [M, N] (temporary: block results)
    block_sums_ptr,     # *fp32, shape [M*B]
    M, N,               # ints
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    num_blocks = (N + BLOCK_N - 1) // BLOCK_N

    for blk in range(0, num_blocks):
        start = blk * BLOCK_N
        running = 0.0

        # Sequential inclusive scan within the block using scalar loads
        for j in range(0, BLOCK_N):
            col = start + j
            # mask: only load if col < N
            if col < N:
                ptr = x_ptr + row * stride_xm + col * stride_xn
                val = tl.load(ptr)
                running = running + val
            # store cumulative
            y_ptr_j = y_ptr + row * stride_ym + col * stride_yn
            tl.store(y_ptr_j, running)

        # Store block sum (last valid cumulative)
        tl.store(block_sums_ptr + row * num_blocks + blk, running)


@triton.jit
def _add_block_offsets_scalar(
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
        # Load block offset
        offset = tl.load(block_prefix_ptr + row * num_blocks + blk)

        # Add offset to each element in the block
        for j in range(0, BLOCK_N):
            col = start + j
            if col < N:
                y_ptr_j = y_ptr + row * stride_ym + col * stride_yn
                val = tl.load(y_ptr_j)
                newVal = val + offset
                tl.store(y_ptr_j, newVal)


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
            x = x.to(torch.float32)

        # Only support 2D and dim=1 for this kernel
        if x.dim() != 2 or self.dim != 1:
            return torch.cumsum(x, dim=self.dim)

        # Make contiguous
        if not x.is_contiguous():
            x = x.contiguous()

        M, N = x.shape
        device = x.device

        # Allocate output and temporaries
        y = torch.empty_like(x)
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
        _block_inclusive_scan_rows_scalar[grid](
            x, y, block_sums,
            M, N,
            stride_xm, stride_xn,
            stride_ym, stride_yn,
            BLOCK_N=BLOCK_N,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # Step 2: exclusive scan of block sums to get block prefixes (Python loop is fine; B is small)
        block_sums_flat = block_sums.view(-1)  # shape [M*B]
        # Compute exclusive prefix on host tensor: p[i] = sum(block_sums_flat[:i])
        # This runs on GPU tensor but is a small O(B) loop; acceptable.
        prefix = torch.empty_like(block_sums_flat)
        running = torch.tensor(0.0, device=device, dtype=torch.float32)
        for i in range(0, block_sums_flat.numel()):
            prefix[i] = running
            running = running + block_sums_flat[i]
        # Write back to block_prefix
        block_prefix_flat = block_prefix.view(-1)
        block_prefix_flat.copy_(prefix)

        # Step 3: add block offsets to y
        _add_block_offsets_scalar[grid](
            y, block_prefix,
            M, N,
            stride_ym, stride_yn,
            BLOCK_N=BLOCK_N,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        return y
