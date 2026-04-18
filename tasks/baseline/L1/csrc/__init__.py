"""Shared CUDA extension loader for L1 baseline kernels.

Compiles and caches the extension once; all task modules import _C from here.
"""

import os

from torch.utils.cpp_extension import load as _load_ext

_DIR = os.path.dirname(os.path.abspath(__file__))

_C = _load_ext(
    name="kb_nano_L1_ops",
    sources=[os.path.join(_DIR, f) for f in [
        "binding.cpp", "rmsnorm.cu", "rmsnorm_quant.cu",
        "activation.cu", "pos_enc.cu",
        "moe_sum.cu", "moe_align.cu", "moe_topk_softmax.cu",
        "eagle_utils.cu",
    ]],
    extra_cuda_cflags=["-O3", "--use_fast_math",
                       "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16"],
    extra_cflags=["-O3"],
    verbose=False,
)
