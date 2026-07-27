I need you to write a high-performance replacement for the following PyTorch nn.Module operator used in an LLM inference engine. Your goal is to produce a kernel that is FASTER than the baseline implementation while remaining numerically correct.

## Baseline implementation

```python
"""Native YOLOv10 detector model."""

from __future__ import annotations

import os
import re

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from ..L2.yolov10_conv import fuse_module
from ..L3.yolov10_backbone import YOLOv10Backbone
from ..L3.yolov10_head import YOLOv10DetectHead
from ..L3.yolov10_neck import YOLOv10Neck

_PREFIX_MAP = [
    ("model.model.0.", "backbone.stem1."),
    ("model.model.1.", "backbone.stem2."),
    ("model.model.2.", "backbone.stage2."),
    ("model.model.3.", "backbone.down3."),
    ("model.model.4.", "backbone.stage3."),
    ("model.model.5.", "backbone.down4."),
    ("model.model.6.", "backbone.stage4."),
    ("model.model.7.", "backbone.down5."),
    ("model.model.8.", "backbone.stage5."),
    ("model.model.9.", "backbone.sppf."),
    ("model.model.10.", "backbone.psa."),
    ("model.model.13.", "neck.c2f_p4."),
    ("model.model.16.", "neck.c2f_p3."),
    ("model.model.17.", "neck.down_p3."),
    ("model.model.19.", "neck.c2f_n4."),
    ("model.model.20.", "neck.down_n4."),
    ("model.model.22.", "neck.c2fcib_n5."),
    ("model.model.23.", "detect."),
]


class YOLOv10ForObjectDetection(nn.Module):
    def __init__(self, conf_threshold: float = 0.25):
        super().__init__()
        self.backbone = YOLOv10Backbone()
        self.neck = YOLOv10Neck()
        self.detect = YOLOv10DetectHead(nc=80, ch=(64, 128, 256))
        self.conf_threshold = conf_threshold

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        conf_threshold: float = 0.25,
    ) -> "YOLOv10ForObjectDetection":
        model = cls(conf_threshold=conf_threshold)
        model_dir = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.bin", "*.json"])
        state_dict = {}
        safetensor_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(safetensor_path):
            state_dict.update(load_file(safetensor_path))
        elif os.path.exists(bin_path):
            loaded = torch.load(bin_path, map_location="cpu")
            state_dict.update(loaded.get("state_dict", loaded))
        else:
            raise FileNotFoundError(f"No YOLOv10 checkpoint found in {model_dir}")

        remapped = {}
        for key, value in state_dict.items():
            replaced = key
            for src, dst in _PREFIX_MAP:
                if key.startswith(src):
                    replaced = dst + key[len(src):]
                    break
            else:
                continue
            remapped[replaced] = value

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        missing = [
            k for k in missing
            if not k.endswith("num_batches_tracked")
            and k not in {"detect.anchors", "detect.strides"}
        ]
        if missing or unexpected:
            raise RuntimeError(f"YOLOv10 weight remap mismatch: missing={missing}, unexpected={unexpected}")

        fuse_module(model)
        model.detect.export = True
        model = model.to(device=device, dtype=dtype).eval()
        return model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values)
        pyramid = self.neck(feats)
        return self.detect(pyramid)

    def predict(
        self,
        pixel_values: torch.Tensor,
        image_size: int,
        max_detections: int = 100,
    ) -> dict[str, torch.Tensor]:
        outputs = self.forward(pixel_values)
        if outputs.shape[-1] != 6:
            raise ValueError(f"Unexpected YOLOv10 export output shape: {tuple(outputs.shape)}")

        boxes = outputs[..., :4]
        scores = outputs[..., 4]
        labels = outputs[..., 5].long()

        if max_detections < boxes.shape[1]:
            boxes = boxes[:, :max_detections]
            scores = scores[:, :max_detections]
            labels = labels[:, :max_detections]

        mask = scores > self.conf_threshold
        batch = scores.shape[0]
        padded_boxes = torch.zeros(batch, max_detections, 4, device=boxes.device)
        padded_scores = torch.zeros(batch, max_detections, device=boxes.device)
        padded_labels = torch.full((batch, max_detections), -1, device=boxes.device, dtype=torch.long)
        for i in range(batch):
            keep = mask[i]
            count = min(int(keep.sum().item()), max_detections)
            if count == 0:
                continue
            padded_boxes[i, :count] = boxes[i, keep][:count]
            padded_scores[i, :count] = scores[i, keep][:count]
            padded_labels[i, :count] = labels[i, keep][:count]

        return {
            "boxes": padded_boxes,
            "scores": padded_scores,
            "labels": padded_labels,
        }

```

## Requirements

1. Your replacement class MUST:
   - Be named `YOLOv10ForObjectDetection` (exactly)
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
- The class definition for `YOLOv10ForObjectDetection`
- Any helper functions, Triton kernels, or CUDA source strings needed

Do NOT include any explanation outside the code block. Do NOT include if __name__ == '__main__' blocks or test code.