"""Linear (matrix multiply) kernel: F.linear(input, weight, bias)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import deep_gemm


class Linear(nn.Module):
    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)


_FP8_INFO = torch.finfo(torch.float8_e4m3fn)
_GROUP_SIZE: tl.constexpr = 128


@triton.jit
def _fp8_group_quant_kernel(
    x_ptr, out_ptr, scale_ptr,
    stride_x_row, stride_out_row, stride_s_row,
    num_cols,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    groups_per_row = num_cols // GROUP_SIZE
    row = pid // groups_per_row
    group = pid % groups_per_row

    x_base = x_ptr + row * stride_x_row + group * GROUP_SIZE
    cols = tl.arange(0, GROUP_SIZE)
    x = tl.load(x_base + cols).to(tl.float32)

    absmax = tl.max(tl.abs(x))
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(absmax / fp8_max)))

    x_scaled = x / scale
    x_clamped = tl.clamp(x_scaled, -fp8_max, fp8_max)
    x_fp8 = x_clamped.to(out_ptr.dtype.element_ty)

    out_base = out_ptr + row * stride_out_row + group * GROUP_SIZE
    tl.store(out_base + cols, x_fp8)

    scale_base = scale_ptr + row * stride_s_row + group
    tl.store(scale_base, scale)


def _per_token_group_quant_fp8(x: torch.Tensor,
                               out_fp8: torch.Tensor,
                               out_scale: torch.Tensor) -> None:
    """In-place per-token-group FP8 quantization with UE8M0 (power-of-two) scales.

    Single Triton kernel launch, writes to pre-allocated buffers for
    CUDA-graph compatibility.
    """
    M, K = x.shape
    groups_per_row = K // _GROUP_SIZE
    _fp8_group_quant_kernel[(M * groups_per_row,)](
        x, out_fp8, out_scale,
        x.stride(0), out_fp8.stride(0), out_scale.stride(0),
        K,
        fp8_max=_FP8_INFO.max,
        GROUP_SIZE=_GROUP_SIZE,
    )


class _Fp8PrefillBufs:
    """Shared prefill buffers for FP8 activation quantization.

    Since decoder layers execute sequentially, a single set of buffers
    (sized for max_num_batched_tokens) can be reused across all Fp8Linear
    instances, eliminating per-layer dynamic allocation during prefill.
    One instance per unique (K, N) weight shape.
    """
    __slots__ = ("a", "s", "o")

    def __init__(self, max_tokens: int, K: int, N: int, device: torch.device):
        num_groups = math.ceil(K / 128)
        self.a = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self.s = torch.empty(max_tokens, num_groups, dtype=torch.float32, device=device)
        self.o = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)


class Fp8Linear(nn.Module):
    """Block-scaled FP8 linear using deep_gemm.fp8_gemm_nt.

    Weights are stored in float8_e4m3fn with pre-processed UE8M0 block scales
    (transformed via deep_gemm.transform_sf_into_required_layout at load time).
    Activations are dynamically quantized to FP8 per-token-group (group=128)
    using in-place ops for CUDA graph compatibility.
    """

    BLOCK_SIZE = 128

    def __init__(self):
        super().__init__()
        self._a_buf: torch.Tensor | None = None
        self._s_buf: torch.Tensor | None = None
        self._o_buf: torch.Tensor | None = None
        self._pf: _Fp8PrefillBufs | None = None

    def _ensure_buffers(self, max_tokens: int, K: int, N: int, device: torch.device):
        """Pre-allocate activation FP8 buffers for CUDA graph capture."""
        num_groups = math.ceil(K / self.BLOCK_SIZE)
        self._a_buf = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self._s_buf = torch.empty(max_tokens, num_groups, dtype=torch.float32, device=device)
        self._o_buf = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)

    def forward(self, input_bf16: torch.Tensor,
                weight_fp8: torch.Tensor,
                weight_scale_inv: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        N, K = weight_fp8.shape
        input_2d = input_bf16.reshape(-1, K)
        M = input_2d.shape[0]

        if self._a_buf is not None and M <= self._a_buf.shape[0]:
            _per_token_group_quant_fp8(input_2d, self._a_buf[:M], self._s_buf[:M])
            q_input = self._a_buf[:M]
            input_scale = self._s_buf[:M]
            output = self._o_buf[:M]
        elif self._pf is not None and M <= self._pf.a.shape[0]:
            _per_token_group_quant_fp8(input_2d, self._pf.a[:M], self._pf.s[:M])
            q_input = self._pf.a[:M]
            input_scale = self._pf.s[:M]
            output = self._pf.o[:M]
        else:
            num_groups = math.ceil(K / self.BLOCK_SIZE)
            q_input = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=input_2d.device)
            input_scale = torch.empty(M, num_groups, dtype=torch.float32, device=input_2d.device)
            _per_token_group_quant_fp8(input_2d, q_input, input_scale)
            output = torch.empty(M, N, dtype=torch.bfloat16, device=input_2d.device)

        deep_gemm.fp8_gemm_nt(
            (q_input, input_scale),
            (weight_fp8, weight_scale_inv),
            output,
        )

        if bias is not None:
            output = output + bias

        return output.view(*input_bf16.shape[:-1], N)


def postprocess_fp8_weights(weight_fp8: torch.Tensor,
                            scale_inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize FP8 weights to UE8M0 scale format and transform scale layout
    for DeepGEMM compatibility. Must be called once after weight loading."""
    N, K = weight_fp8.shape
    block_size = Fp8Linear.BLOCK_SIZE

    scale_rows = math.ceil(N / block_size)
    scale_cols = math.ceil(K / block_size)
    scale = scale_inv[:scale_rows, :scale_cols]

    w_padded = weight_fp8
    need_n_pad = (block_size - N % block_size) % block_size
    need_k_pad = (block_size - K % block_size) % block_size
    if need_n_pad or need_k_pad:
        w_padded = torch.nn.functional.pad(weight_fp8.view(torch.int8),
                                           (0, need_k_pad, 0, need_n_pad)).view(torch.float8_e4m3fn)

    w_view = w_padded.view(math.ceil(N / block_size + need_n_pad / block_size), block_size,
                           math.ceil(K / block_size + need_k_pad / block_size), block_size) \
                      if need_n_pad or need_k_pad else \
                      w_padded.view(scale_rows, block_size, scale_cols, block_size)

    w_bf16 = w_view.to(torch.bfloat16) * scale[:, None, :, None]

    w_bf16_flat = w_bf16.reshape(-1, w_bf16.shape[2] * block_size)
    if need_n_pad or need_k_pad:
        w_bf16_flat = w_bf16_flat[:N, :K].contiguous()

    w_fp8_new, scale_ue8m0 = deep_gemm.per_block_cast_to_fp8(w_bf16_flat, use_ue8m0=True)

    recipe = (1, block_size, block_size)
    scale_transformed = deep_gemm.transform_sf_into_required_layout(
        sf=scale_ue8m0.unsqueeze(0),
        mn=N,
        k=K,
        recipe=recipe,
        num_groups=1,
        is_sfa=False,
    ).squeeze(0)

    return w_fp8_new, scale_transformed
