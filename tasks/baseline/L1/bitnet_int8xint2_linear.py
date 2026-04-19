"""W1.58A8 GEMM: int8 activations × packed-2-bit ternary weights.

This is the L1 primitive that powers ``BitLinear``.  It mirrors Microsoft's
reference ``bitnet_int8xint2_linear`` kernel
(``vllm_repo/BitNet/gpu/bitnet_kernels/bitnet_kernels.cu``) and HF's
``transformers.integrations.bitnet.AutoBitLinear`` forward:

    output_bf16 = (act_int8 @ weight_ternary.T) * weight_scale / act_scale

* ``act_int8``    - per-token int8-quantized activations, shape ``(M, K)``.
* ``act_scale``   - bf16 per-row scale ``s = 127 / max|x|``, shape ``(M,)``.
* ``weight``      - uint8 packed-2bit weights, shape ``(N, K/4)``.  Each byte
                    packs 4 consecutive ternary values for a single output
                    row, biased by +1 (so 0/1/2 -> -1/0/+1).
* ``weight_scale``- bf16 per-output-row dequantization scale, shape ``(N,)``.
                    For non-merged ``BitLinear`` this is the HF scalar broadcast
                    over all rows; for fused ``BitLinearMerged`` it carries
                    the per-shard scale broadcast over each shard's rows.

The ``(N, K/4)`` weight layout is intentionally not the HF on-disk layout
(which is ``(N/4, K)`` packed along OUT).  We re-pack at load time so that
every output row's packed bytes are contiguous along K, making the kernel
trivial: each Triton block reads ``BLOCK_K/4`` bytes per row, expands them
to ``BLOCK_K`` int8 ternary values via bit shifts, and does a standard
``tl.dot`` int8x int8 -> int32 accumulation.

This single kernel handles both ``BitLinear`` (single weight scale broadcast)
and ``BitLinearMerged`` (Q/K/V or gate/up fused) with no kernel changes -
the per-shard scaling is encoded by the ``(N,)`` ``weight_scale`` array.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _bitnet_int8xint2_gemm_kernel(
    A_ptr,            # int8,  (M, K)
    A_scale_ptr,      # bf16,  (M,)
    Wp_ptr,           # uint8, (N, K/4)  -- KN-packed
    Ws_ptr,           # bf16,  (N,)
    Bias_ptr,         # bf16,  (N,) or unused
    Out_ptr,          # bf16,  (M, N)
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,   # must be a multiple of 4
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 4
    K_PACKED = K // 4

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < N

    # Sub-bit shifts within each packed byte: bit slot j -> shift 2*j.
    # offs_k_chunk[k] = (k // 4) * 4 + (k % 4) maps the BLOCK_K dim to
    # ``(packed_byte_idx, sub_bit_idx)`` pairs.  We construct the unpack
    # via 3D broadcast and reshape, which compiles to plain shift+mask.
    offs_kp = tl.arange(0, BLOCK_K_PACKED)            # (BLOCK_K_PACKED,)
    sub = tl.arange(0, 4).to(tl.uint8)                # (4,)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_off in range(0, K_PACKED, BLOCK_K_PACKED):
        k_packed_idx = k_off + offs_kp                # (BLOCK_K_PACKED,)
        k_p_mask = k_packed_idx < K_PACKED            # (BLOCK_K_PACKED,)

        # Load packed weights: (BLOCK_N, BLOCK_K_PACKED) uint8
        wp_ptrs = Wp_ptr + offs_n[:, None] * stride_wn + k_packed_idx[None, :] * stride_wk
        wp = tl.load(
            wp_ptrs,
            mask=n_mask[:, None] & k_p_mask[None, :],
            other=0,
        )                                             # uint8 (BLOCK_N, BLOCK_K_PACKED)

        # Unpack each byte to four int8 ternary values: shape (BLOCK_N, BLOCK_K_PACKED, 4)
        # bit slot j -> ((byte >> 2j) & 3) - 1
        unpacked = ((wp[:, :, None] >> (2 * sub)[None, None, :]) & 3).to(tl.int8) - 1
        # Reshape (BLOCK_N, BLOCK_K_PACKED, 4) -> (BLOCK_N, BLOCK_K) in C order so
        # the rightmost (sub-bit) varies fastest, matching K layout.
        w = tl.reshape(unpacked, (BLOCK_N, BLOCK_K))   # int8 (BLOCK_N, BLOCK_K)

        # Load activations: (BLOCK_M, BLOCK_K) int8.  K indices for this
        # chunk are k_off*4 + [0..BLOCK_K).
        k_idx = k_off * 4 + tl.arange(0, BLOCK_K)      # (BLOCK_K,)
        k_mask = k_idx < K
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
        a = tl.load(
            a_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0,
        )                                              # int8 (BLOCK_M, BLOCK_K)

        # int8 dot product -> int32 accumulator
        # tl.dot expects (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N).
        acc += tl.dot(a, tl.trans(w), out_dtype=tl.int32)

    # Per-row activation scale (BLOCK_M,) and per-row weight scale (BLOCK_N,).
    #
    # The microsoft/bitnet-b1.58-2B-4T checkpoint is consumed by HF's
    # ``transformers.integrations.bitnet.AutoBitLinear`` (NOT the older
    # ``BitLinear`` in the same file - check the model with
    # ``isinstance(model.model.layers[0].self_attn.q_proj, AutoBitLinear)``,
    # confirmed via ``quantization_config.linear_class == 'autobitlinear'``).
    #
    # AutoBitLinear's forward is::
    #
    #     y = F.linear(fake_quant(x), ternary_weight)
    #     y = y * weight_scale
    #
    # where ``fake_quant(x) = round(x * s).clamp(-128, 127) / s`` recovers the
    # original bf16 magnitude.  Equivalently, with explicit int8 quantization::
    #
    #     y = (x_int @ ternary_weight.T) / act_scale * weight_scale
    #
    # i.e. *multiply* by ``weight_scale`` and *divide* by ``act_scale``.  This
    # is the SOTA convention also used by ``vllm_repo/BitNet``'s CUDA kernel
    # (``bitnet_kernels.cu``: ``red_buf0[0] / s[0] * ws[ws_idx]``).
    #
    # Note: the older ``BitLinear`` class in transformers does ``y / (s * ws)``
    # which would be correct if ``weight_scale`` were the *inverse* of the
    # SOTA weight_scale.  That is *not* the case for this checkpoint - the
    # AutoBitLinear path is what was used in training.
    a_scale = tl.load(A_scale_ptr + offs_m, mask=m_mask, other=1.0).to(tl.float32)
    ws = tl.load(Ws_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)

    out_f = acc.to(tl.float32) * ws[None, :] / a_scale[:, None]

    if HAS_BIAS:
        bias = tl.load(Bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        out_f = out_f + bias[None, :]

    out = out_f.to(tl.bfloat16)
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Per-token int8 activation quantization (matches SOTA ``quant_input``):
#
#     s = 127 / max|x|  per row
#     q = round(x * s).clamp(-128, 127)   in int8
#
# ``s`` is returned in bf16 so the GEMM kernel can fold it into the output
# scaling step.
# ---------------------------------------------------------------------------

def _activation_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric int8 activation quantization.

    Mirrors HF's ``transformers.integrations.bitnet.ActQuant.forward``:
    promote to fp32, take per-row absmax (clamped at 1e-5), compute
    ``s = 127 / absmax``, quantize via ``round(x * s).clamp(-128, 127)``.

    Returns ``(q_int8, scale_bf16)`` where ``q_int8`` has the same shape
    as ``x`` and ``scale_bf16`` has shape ``(M,)``.  HF's ``ActQuant``
    keeps the scale in fp32 and recovers a bf16 dequantized activation
    via ``round(x*s)/s``, which rounds away the same precision we would
    by storing ``s`` in bf16.  In practice both choices produce the
    same end-to-end greedy outputs to within ~1 token on this checkpoint;
    bf16 makes the kernel arg slightly cheaper to load.
    """
    x32 = x.to(torch.float32)
    absmax = x32.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5)
    scale = (127.0 / absmax).to(torch.float32)
    q = (x32 * scale).round().clamp_(-128, 127).to(torch.int8)
    return q, scale.squeeze(-1).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# torch.library custom op — opaque to torch.compile (mirrors fp8_linear's
# ``kb_nano_fp8::fp8_gemm_nt`` pattern).
# ---------------------------------------------------------------------------

_bn_lib = torch.library.Library("kb_nano_bitnet", "DEF")

_bn_lib.define(
    "int8xint2_gemm(Tensor input_int8, Tensor act_scale, "
    "Tensor weight, Tensor weight_scale, Tensor? bias, Tensor! output) -> ()"
)


def _bitnet_gemm_impl(
    input_int8: torch.Tensor,
    act_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    M, K = input_int8.shape
    N, K_packed = weight.shape
    assert K == K_packed * 4, f"K mismatch: input K={K} vs weight K/4={K_packed}"
    assert weight_scale.numel() == N
    assert act_scale.numel() == M

    # Pick block sizes.  H100/H200 has plenty of SMs; small N/K can fit in
    # one block per dim.  Triton's int8 ``tl.dot`` requires BLOCK_M>=16.
    BLOCK_M = 16 if M >= 16 else triton.next_power_of_2(max(M, 16))
    BLOCK_M = min(BLOCK_M, 64)
    BLOCK_N = 64
    BLOCK_K = 64
    if BLOCK_K > K:
        BLOCK_K = max(4, triton.next_power_of_2(K) // 1)
        BLOCK_K = ((BLOCK_K + 3) // 4) * 4

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    bias_ptr = bias if bias is not None else input_int8  # dummy ptr; HAS_BIAS gates loads
    _bitnet_int8xint2_gemm_kernel[grid](
        input_int8, act_scale, weight, weight_scale, bias_ptr, output,
        M, N, K,
        input_int8.stride(0), input_int8.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )


_bn_lib.impl("int8xint2_gemm", _bitnet_gemm_impl, "CUDA")


@torch.library.impl(_bn_lib, "int8xint2_gemm", "Meta")
def _bitnet_gemm_meta(input_int8, act_scale, weight, weight_scale, bias, output):
    pass


# ---------------------------------------------------------------------------
# Public entry point used by ``BitLinear`` / ``BitLinearMerged``.
# ---------------------------------------------------------------------------

def bitnet_int8xint2_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """W1.58A8 linear: bf16 activation -> int8 quant -> int8xint2 GEMM -> bf16.

    Args:
        x: ``(..., K)`` bf16/fp16 activation.
        weight: ``(N, K/4)`` uint8 packed ternary weight (KN-packed layout).
        weight_scale: ``(N,)`` bf16 per-output-row dequant scale.
        bias: optional ``(N,)`` bf16 bias added in fp32 inside the kernel.

    Returns:
        ``(..., N)`` bf16 output.
    """
    assert weight.dtype == torch.uint8
    assert weight.ndim == 2
    N, K_packed = weight.shape
    K = K_packed * 4
    in_shape = x.shape
    assert in_shape[-1] == K, f"x last dim {in_shape[-1]} != K {K}"

    x_2d = x.reshape(-1, K).contiguous()
    M = x_2d.shape[0]

    q_int8, act_scale = _activation_quant_int8(x_2d)

    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    torch.ops.kb_nano_bitnet.int8xint2_gemm(
        q_int8, act_scale, weight, weight_scale, bias, out,
    )

    return out.reshape(*in_shape[:-1], N)


# ---------------------------------------------------------------------------
# Weight repacking: HF (out/4, in) uint8 -> KN-packed (out, in/4) uint8
# ---------------------------------------------------------------------------

VALUES_PER_BYTE = 4


def unpack_hf_ternary(packed_hf: torch.Tensor) -> torch.Tensor:
    """Unpack HF's (out/4, in) uint8 into int8 (out, in) ternary in {-1,0,+1}.

    Mirrors ``transformers.integrations.bitnet.unpack_weights``: byte at
    (r, c) holds 4 ternary values for output rows ``r + i*(out/4)`` for
    ``i in [0, 1, 2, 3]`` in bits ``2i .. 2i+1``, biased by +1.
    """
    if packed_hf.dtype != torch.uint8:
        packed_hf = packed_hf.to(torch.uint8)
    packed_rows, in_features = packed_hf.shape
    out_rows = packed_rows * VALUES_PER_BYTE
    unpacked = torch.empty((out_rows, in_features), dtype=torch.uint8,
                           device=packed_hf.device)
    for i in range(VALUES_PER_BYTE):
        mask = 3 << (2 * i)
        unpacked[i * packed_rows:(i + 1) * packed_rows] = (packed_hf & mask) >> (2 * i)
    return unpacked.to(torch.int8) - 1


def repack_ternary_kn(unpacked: torch.Tensor) -> torch.Tensor:
    """Repack int8 ternary ``(out, in)`` into uint8 ``(out, in/4)`` (KN layout).

    Each output byte stores 4 consecutive ternary K values for one output
    row, biased by +1.  Bit ``2j..2j+1`` of byte ``(n, c)`` holds the value
    at ``(n, 4c + j)``.
    """
    assert unpacked.dim() == 2
    out_features, in_features = unpacked.shape
    assert in_features % VALUES_PER_BYTE == 0, \
        f"in_features={in_features} not divisible by {VALUES_PER_BYTE}"
    biased = (unpacked + 1).to(torch.int32)            # (out, in), {0,1,2}
    biased = biased.reshape(out_features, in_features // VALUES_PER_BYTE,
                            VALUES_PER_BYTE)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int32,
                          device=unpacked.device)
    packed = (biased << shifts).sum(dim=-1).to(torch.uint8)
    return packed.contiguous()


def hf_packed_to_kn_packed(packed_hf: torch.Tensor) -> torch.Tensor:
    """Convenience: HF ``(out/4, in)`` -> KN-packed ``(out, in/4)`` uint8."""
    return repack_ternary_kn(unpack_hf_ternary(packed_hf))


# ---------------------------------------------------------------------------
# Module wrapper for use as an L1 op in benchmarks / kernel-swapper plumbing.
# ---------------------------------------------------------------------------

class BitnetInt8xInt2Gemm(nn.Module):
    """Stateless wrapper around the W1.58A8 GEMM custom op."""

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                weight_scale: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        return bitnet_int8xint2_linear(x, weight, weight_scale, bias)
