I need you to write a high-performance replacement for the following PyTorch nn.Module operator used in an LLM inference engine. Your goal is to produce a kernel that is FASTER than the baseline implementation while remaining numerically correct.

## Baseline implementation

```python
"""YOLOv10 native neck."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.interpolate import Interpolate
from ..L2.yolov10_c2f import YOLOC2f, YOLOC2fCIB
from ..L2.yolov10_concat import YOLOConcat
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_scdown import YOLOSCDown


class YOLOv10Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self._upsample = Interpolate()
        self.cat1 = YOLOConcat(1)
        self.c2f_p4 = YOLOC2f(384, 128, n=1, shortcut=False)
        self.cat2 = YOLOConcat(1)
        self.c2f_p3 = YOLOC2f(192, 64, n=1, shortcut=False)
        self.down_p3 = YOLOConv(64, 64, 3, 2)
        self.cat3 = YOLOConcat(1)
        self.c2f_n4 = YOLOC2f(192, 128, n=1, shortcut=False)
        self.down_n4 = YOLOSCDown(128, 128, 3, 2)
        self.cat4 = YOLOConcat(1)
        self.c2fcib_n5 = YOLOC2fCIB(384, 256, n=1, shortcut=True, lk=True)

    def forward(self, feats: dict[str, torch.Tensor]):
        p3_backbone = feats["p3_backbone"]
        p4_backbone = feats["p4_backbone"]
        p5_backbone = feats["p5_backbone"]

        x = self._upsample(p5_backbone, scale_factor=2.0, mode="nearest")
        x = self.cat1([x, p4_backbone])
        p4 = self.c2f_p4(x)

        x = self._upsample(p4, scale_factor=2.0, mode="nearest")
        x = self.cat2([x, p3_backbone])
        p3 = self.c2f_p3(x)

        x = self.down_p3(p3)
        x = self.cat3([x, p4])
        n4 = self.c2f_n4(x)

        x = self.down_n4(n4)
        x = self.cat4([x, p5_backbone])
        n5 = self.c2fcib_n5(x)
        return [p3, n4, n5]

```

## Requirements

1. Your replacement class MUST:
   - Be named `YOLOv10Neck` (exactly)
   - Subclass `torch.nn.Module`
   - Have the EXACT same `forward` signature (same parameter names, types, defaults)
   - Produce numerically equivalent outputs (or very close)
   - The `__init__` method is optional -- you only need to override it if you need to change initialization logic (e.g. pre-allocate buffers). If you do override it, keep the same signature.
2. You may use Triton, PyTorch, raw CUDA, or any combination. Aim for the highest performance possible on NVIDIA H200 GPUs.
3. Do NOT import `vllm`, `sglang`, or `sgl_kernel`.
3b. Your file is imported STANDALONE, outside the package, so RELATIVE imports (`from ..L1.x import Y`) will fail with ImportError. To reuse a baseline component, import it absolutely as `from fastkernels.tasks.baseline.L<n>.<module> import <Class>` (e.g. `from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm`), or inline the code you need. Never emit a relative import.
4. You may import `torch`, `triton`, `triton.language`, standard library modules, `flash_attn`, or JIT-compile CUDA. For inline CUDA strings use `torch.utils.cpp_extension.load_inline(name=..., cpp_sources=..., cuda_sources=..., functions=[...], build_directory='/mnt/weka/home/hao.zhang/async_rl_bench/kb_nano/agent/_cuda_build_cache/<unique_name>')`. Do NOT pass `cuda_sources` to `torch.utils.cpp_extension.load()` (it only takes file paths via `sources`).
5. Focus on PERFORMANCE: minimize memory traffic, maximize GPU occupancy, fuse operations where possible, and use vectorized loads/stores.

## Response format

Return ONLY a single Python code block (```python ... ```) containing:
- All necessary imports at the top
- The class definition for `YOLOv10Neck`
- Any helper functions, Triton kernels, or CUDA source strings needed

Do NOT include any explanation outside the code block. Do NOT include if __name__ == '__main__' blocks or test code.