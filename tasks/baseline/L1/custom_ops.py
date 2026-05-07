"""Register opaque kernels as torch.library custom ops for torch.compile.

When torch.compile traces a decoder layer, it needs fake (meta) tensor
implementations for operations it cannot inline.  Operations that use
pure PyTorch (RMSNorm-native, SiLU-native) are inlined and fused by
Inductor; everything else is registered here as an opaque custom op.

Import this module once at startup to populate torch.ops.kb_nano.*.
"""

from __future__ import annotations

import torch
from torch.library import Library, infer_schema

kb_lib = Library("kb_nano", "FRAGMENT")


def _register(name, func, mutates_args=None, fake_impl=None,
              dispatch_key="CUDA"):
    if mutates_args is None:
        mutates_args = []
    schema = infer_schema(func, mutates_args=mutates_args)
    kb_lib.define(name + schema)
    kb_lib.impl(name, func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        kb_lib._register_fake(name, fake_impl)


# ---- FP8 block-scaled GEMM (DeepGEMM) ----------------------------------

def _deep_gemm_fp8_nt(
    A: torch.Tensor, As: torch.Tensor,
    B: torch.Tensor, Bs: torch.Tensor,
    output: torch.Tensor,
    use_e8m0: bool,
) -> None:
    import deep_gemm
    kwargs = {} if use_e8m0 else {"disable_ue8m0_cast": True}
    deep_gemm.fp8_gemm_nt((A, As), (B, Bs), output, **kwargs)


def _deep_gemm_fp8_nt_fake(
    A: torch.Tensor, As: torch.Tensor,
    B: torch.Tensor, Bs: torch.Tensor,
    output: torch.Tensor,
    use_e8m0: bool,
) -> None:
    return None


_register("deep_gemm_fp8_nt", _deep_gemm_fp8_nt,
          mutates_args=["output"], fake_impl=_deep_gemm_fp8_nt_fake)


# ---- FP8 block-scaled GEMM (Triton fallback) ----------------------------

def _triton_block_scaled_mm(
    A: torch.Tensor, B: torch.Tensor,
    As: torch.Tensor, Bs: torch.Tensor,
    output: torch.Tensor,
    block_n: int, block_k: int,
) -> None:
    from .fp8_block_scaled_mm import _w8a8_block_scaled_mm_kernel
    import triton
    M = A.shape[0]
    N, K = B.shape
    config = {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k, "GROUP_SIZE_M": 32,
        "num_warps": 4, "num_stages": 2,
    }

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    _w8a8_block_scaled_mm_kernel[grid](
        A, B, output, As, Bs,
        M, N, K, block_n, block_k,
        A.stride(-2), A.stride(-1),
        B.stride(1), B.stride(0),
        output.stride(0), output.stride(1),
        As.stride(-2), As.stride(-1),
        Bs.stride(1), Bs.stride(0),
        **config,
    )


def _triton_block_scaled_mm_fake(
    A: torch.Tensor, B: torch.Tensor,
    As: torch.Tensor, Bs: torch.Tensor,
    output: torch.Tensor,
    block_n: int, block_k: int,
) -> None:
    return None


_register("triton_block_scaled_mm", _triton_block_scaled_mm,
          mutates_args=["output"],
          fake_impl=_triton_block_scaled_mm_fake)


# ---- Per-token-group FP8 quantization -----------------------------------

def _per_token_group_quant_fp8(
    x: torch.Tensor,
    q_buf: torch.Tensor,
    s_buf: torch.Tensor,
    group_size: int,
    use_ue8m0: bool,
) -> None:
    """Quantize x into q_buf/s_buf in-place."""
    from .fp8_quant import (
        _HAS_VLLM_CUDA_QUANT, _FP8_MIN, _FP8_MAX,
        _per_token_group_quant_fp8_kernel,
    )
    import triton

    if _HAS_VLLM_CUDA_QUANT:
        torch.ops._C.per_token_group_fp8_quant(
            x, q_buf, s_buf,
            group_size, 1e-10,
            _FP8_MIN, _FP8_MAX, use_ue8m0,
        )
        return

    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    _per_token_group_quant_fp8_kernel[(M,)](
        x, q_buf, s_buf,
        group_size, x.shape[-1], x.stride(-2), 1e-10,
        fp8_min=_FP8_MIN, fp8_max=_FP8_MAX,
        USE_UE8M0=use_ue8m0,
        BLOCK=BLOCK, num_warps=num_warps, num_stages=1,
    )


def _per_token_group_quant_fp8_fake(
    x: torch.Tensor,
    q_buf: torch.Tensor,
    s_buf: torch.Tensor,
    group_size: int,
    use_ue8m0: bool,
) -> None:
    return None


_register("per_token_group_quant_fp8", _per_token_group_quant_fp8,
          mutates_args=["q_buf", "s_buf"],
          fake_impl=_per_token_group_quant_fp8_fake)


# ---- SiLU-and-mul -------------------------------------------------------

def _silu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    from sgl_kernel import silu_and_mul as _sgl_silu_and_mul
    _sgl_silu_and_mul(x, out)


def _silu_and_mul_fake(x: torch.Tensor, out: torch.Tensor) -> None:
    return None


_register("silu_and_mul", _silu_and_mul,
          mutates_args=["out"], fake_impl=_silu_and_mul_fake)


# ---- RoPE in-place (sgl_kernel) -----------------------------------------

def _rope_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    cos_sin_cache: torch.Tensor,
) -> None:
    from sgl_kernel import apply_rope_with_cos_sin_cache_inplace as _sgl_rope
    _sgl_rope(positions, query, key, head_dim, cos_sin_cache)


def _rope_inplace_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    cos_sin_cache: torch.Tensor,
) -> None:
    return None


_register("rope_inplace", _rope_inplace,
          mutates_args=["query", "key"], fake_impl=_rope_inplace_fake)


# ---- M-RoPE in-place (Triton kernel) ------------------------------------

def _mrope_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_3d: torch.Tensor,
    sin_3d: torch.Tensor,
    num_tokens: int,
    n_qh: int, n_kh: int,
    hd: int,
    pad_n_qh: int, pad_n_kh: int, pad_hd: int,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
) -> None:
    from .mrope import _mrope_kernel
    _mrope_kernel[(num_tokens,)](
        query, key, cos_3d, sin_3d,
        num_tokens, n_qh, n_kh, hd, hd,
        pad_n_qh, pad_n_kh, pad_hd,
        mrope_section_t, mrope_section_h, mrope_section_w,
        is_interleaved,
    )


def _mrope_inplace_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_3d: torch.Tensor,
    sin_3d: torch.Tensor,
    num_tokens: int,
    n_qh: int, n_kh: int,
    hd: int,
    pad_n_qh: int, pad_n_kh: int, pad_hd: int,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
) -> None:
    return None


_register("mrope_inplace", _mrope_inplace,
          mutates_args=["query", "key"], fake_impl=_mrope_inplace_fake)


# ---- Store KV cache (Triton) --------------------------------------------

def _store_kvcache(
    key: torch.Tensor, value: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    from .store_kvcache import _store_kvcache_kernel
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    _store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slot_mapping, D,
    )


def _store_kvcache_fake(
    key: torch.Tensor, value: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    return None


_register("store_kvcache", _store_kvcache,
          mutates_args=["k_cache", "v_cache"],
          fake_impl=_store_kvcache_fake)


def _store_kvcache_hnd(
    key: torch.Tensor, value: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_size: int,
) -> None:
    from .store_kvcache import _store_kvcache_hnd_kernel
    N, num_kv_heads, head_dim = key.shape
    _store_kvcache_hnd_kernel[(N, num_kv_heads)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slot_mapping,
        PAGE_SIZE=page_size,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
    )


def _store_kvcache_hnd_fake(
    key: torch.Tensor, value: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_size: int,
) -> None:
    return None


_register("store_kvcache_hnd", _store_kvcache_hnd,
          mutates_args=["k_cache", "v_cache"],
          fake_impl=_store_kvcache_hnd_fake)


# ---- Flash Attention Decode ----------------------------------------------

def _flash_attn_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    max_seq_len: int,
) -> torch.Tensor:
    from flash_attn import flash_attn_with_kvcache
    return flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=True,
    ).squeeze(1)


def _flash_attn_decode_fake(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    max_seq_len: int,
) -> torch.Tensor:
    return torch.empty_like(q)


_register("flash_attn_decode", _flash_attn_decode,
          fake_impl=_flash_attn_decode_fake)


# ---- Flash Attention Prefill ---------------------------------------------

def _flash_attn_prefill(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int, max_seqlen_k: int,
    softmax_scale: float,
) -> torch.Tensor:
    from flash_attn import flash_attn_varlen_func
    return flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
    )


def _flash_attn_prefill_fake(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int, max_seqlen_k: int,
    softmax_scale: float,
) -> torch.Tensor:
    return torch.empty_like(q)


_register("flash_attn_prefill", _flash_attn_prefill,
          fake_impl=_flash_attn_prefill_fake)


# ---- AllReduce -----------------------------------------------------------

def _allreduce(tensor: torch.Tensor) -> torch.Tensor:
    ar = _get_custom_ar_for_op()
    if ar is not None:
        out = ar.custom_all_reduce(tensor)
        if out is not None:
            return out
    torch.distributed.all_reduce(tensor)
    return tensor


def _allreduce_fake(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor)


def _get_custom_ar_for_op():
    from .allreduce import get_custom_ar
    return get_custom_ar()


_register("allreduce", _allreduce, fake_impl=_allreduce_fake)


# ---- RMSNorm (Triton) ---------------------------------------------------

def _rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from .rms_norm import _rmsnorm as _rmsnorm_impl
    return _rmsnorm_impl(x, weight, eps)


def _rmsnorm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


_register("rmsnorm", _rmsnorm, fake_impl=_rmsnorm_fake)


def _fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    from .rms_norm import _fused_add_rmsnorm as _fused_add_rmsnorm_impl
    _fused_add_rmsnorm_impl(x, residual, weight, eps)


def _fused_add_rmsnorm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    return None


_register("fused_add_rmsnorm", _fused_add_rmsnorm,
          mutates_args=["x", "residual"],
          fake_impl=_fused_add_rmsnorm_fake)
