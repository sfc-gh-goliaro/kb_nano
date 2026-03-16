"""Block-scaled FP8 GEMM: DeepGEMM (preferred) with Triton fallback.

Computes C = A @ B^T where A and B are in float8_e4m3fn with per-block
scale factors. Uses DeepGEMM on Blackwell/Hopper GPUs for maximum
throughput, falling back to a Triton kernel otherwise.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

_HAS_DEEP_GEMM = False
_deep_gemm = None
_DEEP_GEMM_E8M0 = False
try:
    import deep_gemm as _deep_gemm
    _HAS_DEEP_GEMM = hasattr(_deep_gemm, "fp8_gemm_nt")
    if _HAS_DEEP_GEMM and torch.cuda.is_available():
        _DEEP_GEMM_E8M0 = torch.cuda.get_device_capability(0)[0] >= 10
except ImportError:
    pass


@triton.jit
def _w8a8_block_scaled_mm_kernel(
    A, B, C, As, Bs,
    M, N, K,
    group_n, group_k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_As_m, stride_As_k,
    stride_Bs_k, stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


class W8A8BlockScaledMM(nn.Module):
    """Block-scaled FP8 matrix multiplication.

    forward(A, B, As, Bs, block_size, output_dtype) -> C:
        A:  [M, K] float8_e4m3fn (activation, contiguous)
        B:  [N, K] float8_e4m3fn (weight, stored as [N, K])
        As: [M, ceil(K/bk)] float32 (activation scales)
        Bs: [ceil(N/bn), ceil(K/bk)] float32 (weight scales)
        block_size: (bn, bk) quantization block dimensions
        output_dtype: dtype for output tensor C (default bfloat16)
        C:  [M, N] in output_dtype
    """

    def __init__(self):
        super().__init__()
        self._out_buf: torch.Tensor | None = None

    def _get_output(self, M: int, N: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        needed = M * N
        buf = self._out_buf
        if buf is None or buf.numel() < needed:
            buf = torch.empty(needed, dtype=dtype, device=device)
            self._out_buf = buf
        return buf[:needed].view(M, N)

    def set_shared_buffer(self, buf: torch.Tensor):
        self._out_buf = buf.view(-1) if buf.ndim > 1 else buf

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        block_size: tuple[int, int],
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        assert A.shape[-1] == B.shape[-1]
        assert A.is_contiguous()
        M = A.numel() // A.shape[-1]
        N, K = B.shape

        C = self._get_output(M, N, output_dtype, A.device)

        if _HAS_DEEP_GEMM and output_dtype == torch.bfloat16 and N % 64 == 0 and K % 128 == 0:
            kwargs = {} if _DEEP_GEMM_E8M0 else {"disable_ue8m0_cast": True}
            _deep_gemm.fp8_gemm_nt(
                (A.view(M, K), As.view(M, -1)),
                (B, Bs),
                C,
                **kwargs,
            )
            return C

        block_n, block_k = block_size

        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 2,
        }

        def grid(META):
            return (
                triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            )

        _w8a8_block_scaled_mm_kernel[grid](
            A, B, C, As, Bs,
            M, N, K,
            block_n, block_k,
            A.stride(-2), A.stride(-1),
            B.stride(1), B.stride(0),
            C.stride(0), C.stride(1),
            As.stride(-2), As.stride(-1),
            Bs.stride(1), Bs.stride(0),
            **config,
        )

        return C
