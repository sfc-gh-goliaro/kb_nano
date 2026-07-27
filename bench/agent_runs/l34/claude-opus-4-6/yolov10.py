from __future__ import annotations

import os
import types

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from fastkernels.tasks.baseline.L2.yolov10_conv import fuse_module
from fastkernels.tasks.baseline.L3.yolov10_backbone import YOLOv10Backbone
from fastkernels.tasks.baseline.L3.yolov10_head import (
    YOLOv10DetectHead,
    make_anchors,
    dist2bbox,
)
from fastkernels.tasks.baseline.L3.yolov10_neck import YOLOv10Neck


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


def _patched_attn_forward(self, x):
    b, c, h, w = x.shape
    n = h * w
    qkv = self.qkv(x)
    q, k, v = qkv.reshape(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
        [self.key_dim, self.key_dim, self.head_dim], dim=2
    )
    attn = (q.transpose(-2, -1) @ k) * self.scale
    attn = self._softmax(attn)
    x = (v @ attn.transpose(-2, -1)).reshape(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
    return self.proj(x)


def _patched_detect_inference(self, x):
    shape = x[0].shape
    x_cat = torch.cat([xi.reshape(shape[0], self.no, -1) for xi in x], 2)
    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (
            t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5)
        )
        self.shape = shape
    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    dbox = (
        dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1)
        * self.strides
    )
    return torch.cat((dbox, self._sigmoid(cls)), 1)


class YOLOv10ForObjectDetection(nn.Module):
    def __init__(self, conf_threshold: float = 0.25):
        super().__init__()
        self.backbone = YOLOv10Backbone()
        self.neck = YOLOv10Neck()
        self.detect = YOLOv10DetectHead(nc=80, ch=(64, 128, 256))
        self.conf_threshold = conf_threshold
        self._cuda_graphs = {}

    def _raw_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values)
        pyramid = self.neck(feats)
        return self.detect(pyramid)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        conf_threshold: float = 0.25,
    ) -> "YOLOv10ForObjectDetection":
        model = cls(conf_threshold=conf_threshold)
        model_dir = snapshot_download(
            model_name,
            allow_patterns=["*.safetensors", "*.bin", "*.json"],
        )
        state_dict = {}
        safetensor_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(safetensor_path):
            state_dict.update(load_file(safetensor_path))
        elif os.path.exists(bin_path):
            loaded = torch.load(bin_path, map_location="cpu")
            state_dict.update(loaded.get("state_dict", loaded))
        else:
            raise FileNotFoundError(
                f"No YOLOv10 checkpoint found in {model_dir}"
            )

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
            k
            for k in missing
            if not k.endswith("num_batches_tracked")
            and k not in {"detect.anchors", "detect.strides"}
        ]
        if missing or unexpected:
            raise RuntimeError(
                f"YOLOv10 weight remap mismatch: missing={missing}, unexpected={unexpected}"
            )

        fuse_module(model)
        model.detect.export = True
        model = model.to(device=device, dtype=dtype).eval()

        torch.backends.cudnn.benchmark = True

        model.backbone.psa.attn.forward = types.MethodType(
            _patched_attn_forward, model.backbone.psa.attn
        )
        model.detect.inference = types.MethodType(
            _patched_detect_inference, model.detect
        )

        model = model.to(memory_format=torch.channels_last)

        is_cuda = "cuda" in str(device)

        with torch.no_grad():
            dummy = torch.randn(
                1, 3, 640, 640, device=device, dtype=dtype
            ).contiguous(memory_format=torch.channels_last)
            for _ in range(3):
                model._raw_forward(dummy)

        if is_cuda:
            try:
                shape = (1, 3, 640, 640)
                static_input = torch.empty(
                    shape, device=device, dtype=dtype
                ).contiguous(memory_format=torch.channels_last)

                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        model._raw_forward(static_input)
                torch.cuda.current_stream().wait_stream(s)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_output = model._raw_forward(static_input)

                model._cuda_graphs[shape] = (graph, static_input, static_output)
            except Exception:
                model._cuda_graphs.clear()

        return model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        key = tuple(pixel_values.shape)
        entry = self._cuda_graphs.get(key)
        if entry is not None:
            graph, static_input, static_output = entry
            static_input.copy_(pixel_values)
            graph.replay()
            return static_output.clone()
        return self._raw_forward(pixel_values)

    def predict(
        self,
        pixel_values: torch.Tensor,
        image_size: int,
        max_detections: int = 100,
    ) -> dict[str, torch.Tensor]:
        outputs = self.forward(pixel_values)
        if outputs.shape[-1] != 6:
            raise ValueError(
                f"Unexpected YOLOv10 export output shape: {tuple(outputs.shape)}"
            )

        boxes = outputs[..., :4]
        scores = outputs[..., 4]
        labels = outputs[..., 5].long()

        if max_detections < boxes.shape[1]:
            boxes = boxes[:, :max_detections]
            scores = scores[:, :max_detections]
            labels = labels[:, :max_detections]

        mask = scores > self.conf_threshold
        batch = scores.shape[0]
        counts = mask.sum(dim=1).clamp(max=max_detections)

        sorted_idx = mask.long().argsort(dim=1, descending=True, stable=True)[
            :, :max_detections
        ]

        gathered_boxes = boxes.gather(
            1, sorted_idx.unsqueeze(-1).expand(-1, -1, 4)
        )
        gathered_scores = scores.gather(1, sorted_idx)
        gathered_labels = labels.gather(1, sorted_idx)

        arange = torch.arange(max_detections, device=boxes.device).unsqueeze(0)
        valid = arange < counts.unsqueeze(1)

        padded_boxes = torch.zeros(
            batch, max_detections, 4, device=boxes.device
        )
        padded_scores = torch.zeros(
            batch, max_detections, device=boxes.device
        )
        padded_labels = torch.full(
            (batch, max_detections), -1, device=boxes.device, dtype=torch.long
        )

        padded_boxes[valid] = gathered_boxes[valid]
        padded_scores[valid] = gathered_scores[valid]
        padded_labels[valid] = gathered_labels[valid]

        return {
            "boxes": padded_boxes,
            "scores": padded_scores,
            "labels": padded_labels,
        }
