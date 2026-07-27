I need you to write a high-performance replacement for the following PyTorch nn.Module operator used in an LLM inference engine. Your goal is to produce a kernel that is FASTER than the baseline implementation while remaining numerically correct.

## Baseline implementation

```python
"""YOLOv10 native backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L2.yolov10_c2f import YOLOC2f
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_psa import YOLOPSA
from ..L2.yolov10_scdown import YOLOSCDown
from ..L2.yolov10_sppf import YOLOSPPF


class YOLOv10Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem1 = YOLOConv(3, 16, 3, 2)
        self.stem2 = YOLOConv(16, 32, 3, 2)
        self.stage2 = YOLOC2f(32, 32, n=1, shortcut=True)
        self.down3 = YOLOConv(32, 64, 3, 2)
        self.stage3 = YOLOC2f(64, 64, n=2, shortcut=True)
        self.down4 = YOLOSCDown(64, 128, 3, 2)
        self.stage4 = YOLOC2f(128, 128, n=2, shortcut=True)
        self.down5 = YOLOSCDown(128, 256, 3, 2)
        self.stage5 = YOLOC2f(256, 256, n=1, shortcut=True)
        self.sppf = YOLOSPPF(256, 256, 5)
        self.psa = YOLOPSA(256, 256)

    def forward(self, x: torch.Tensor):
        x = self.stem1(x)
        x = self.stem2(x)
        p2 = self.stage2(x)
        x = self.down3(p2)
        p3 = self.stage3(x)
        x = self.down4(p3)
        p4 = self.stage4(x)
        x = self.down5(p4)
        p5 = self.stage5(x)
        p5 = self.sppf(p5)
        p5 = self.psa(p5)
        return {"p3_backbone": p3, "p4_backbone": p4, "p5_backbone": p5}

```

## Requirements

1. Your replacement class MUST:
   - Be named `YOLOv10Backbone` (exactly)
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
- The class definition for `YOLOv10Backbone`
- Any helper functions, Triton kernels, or CUDA source strings needed

Do NOT include any explanation outside the code block. Do NOT include if __name__ == '__main__' blocks or test code.