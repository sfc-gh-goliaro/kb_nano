"""Shared CUDA extension loader for L1 baseline kernels.

Compiles and caches the extension once; all task modules import _C from here.
"""

import glob
import os

import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import load as _load_ext

_DIR = os.path.dirname(os.path.abspath(__file__))


def _toolkit_major(cuda_home: str) -> str | None:
    """CUDA major version of a toolkit, from the libcudart soname it ships."""
    for pattern in ("targets/*/lib/libcudart.so.*", "lib64/libcudart.so.*"):
        for so in glob.glob(os.path.join(cuda_home, pattern)):
            major = so.rsplit(".so.", 1)[-1].split(".")[0]
            if major.isdigit():
                return major
    return None


def _cuda_home_matching_torch() -> str | None:
    """Pick a CUDA toolkit whose major version matches torch's build.

    This extension links ``-lcublas``.  When the system-default toolkit is a
    different CUDA major than torch (e.g. ``/usr/local/cuda`` -> 13.0 while
    torch is 2.10.0+cu128, which loads cuBLAS 12 from the nvidia wheels), the
    extension resolves cuBLAS 13 while ``at::cuda::getCurrentCUDABlasHandle()``
    returns a cuBLAS-12 handle.  cuBLAS 13 then rejects that handle and the
    first router GEMM dies with ``CUBLAS_STATUS_NOT_INITIALIZED`` -- which is
    how Kimi-Linear failed on B200, where 13.0 is the default toolkit.
    """
    want = (torch.version.cuda or "").split(".")[0]
    if not want:
        return cpp_extension.CUDA_HOME
    current = cpp_extension.CUDA_HOME
    if current and _toolkit_major(current) == want:
        return current
    # Prefer the highest matching minor version (12.9 over 12.0).
    for cand in sorted(glob.glob("/usr/local/cuda-*"), reverse=True):
        if os.path.isdir(cand) and _toolkit_major(cand) == want:
            return cand
    return current


_CUDA_HOME = _cuda_home_matching_torch()
if _CUDA_HOME and _CUDA_HOME != cpp_extension.CUDA_HOME:
    # cpp_extension resolves CUDA_HOME at import time; override both so nvcc and
    # the -L/-lcublas search path agree.
    cpp_extension.CUDA_HOME = _CUDA_HOME
    os.environ["CUDA_HOME"] = _CUDA_HOME

_C = _load_ext(
    name="fastkernels_L1_ops",
    sources=[os.path.join(_DIR, f) for f in [
        "binding.cpp", "rmsnorm.cu", "rmsnorm_quant.cu",
        "activation.cu", "pos_enc.cu",
        "moe_sum.cu", "moe_align.cu", "moe_topk_softmax.cu",
        "eagle_utils.cu",
        # DeepSeek-V3 router ops (verbatim port of vLLM csrc/moe sources;
        # see binding.cpp for op-level descriptions).
        "dsv3_router_gemm_entry.cu",
        "dsv3_router_gemm_float_out.cu",
        "dsv3_router_gemm_bf16_out.cu",
        "router_gemm_bf16_fp32.cu",
        "grouped_topk_kernels.cu",
    ]],
    extra_cuda_cflags=["-O3",
                       "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16",
                       # vLLM's CMake unsets these so its noaux_tc grouped-topk
                       # kernel (ported verbatim into ``grouped_topk_kernels.cu``)
                       # can rely on implicit ``half``/``__nv_bfloat16``<->``float``
                       # constructors.  ``torch.utils.cpp_extension`` defines them
                       # by default; we undefine them here to match vLLM.
                       "-U__CUDA_NO_HALF_OPERATORS__",
                       "-U__CUDA_NO_HALF_CONVERSIONS__",
                       "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                       "-U__CUDA_NO_HALF2_OPERATORS__"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
    verbose=False,
)
