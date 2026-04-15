"""Native YOLOv10 detector model."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..L2.yolov10_conv import fuse_module
from ..L3.yolov10_backbone import YOLOv10Backbone
from ..L3.yolov10_head import YOLOv10DetectHead
from ..L3.yolov10_neck import YOLOv10Neck


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_yolov10_repo_on_path() -> None:
    repo_root = _repo_root() / "third_party" / "yolov10"
    if not repo_root.exists():
        raise FileNotFoundError(
            f"Missing YOLOv10 repo at {repo_root}. "
            "Clone https://github.com/THU-MIG/yolov10 into third_party/yolov10."
        )
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


_PREFIX_MAP = [
    ("model.0.", "backbone.stem1."),
    ("model.1.", "backbone.stem2."),
    ("model.2.", "backbone.stage2."),
    ("model.3.", "backbone.down3."),
    ("model.4.", "backbone.stage3."),
    ("model.5.", "backbone.down4."),
    ("model.6.", "backbone.stage4."),
    ("model.7.", "backbone.down5."),
    ("model.8.", "backbone.stage5."),
    ("model.9.", "backbone.sppf."),
    ("model.10.", "backbone.psa."),
    ("model.13.", "neck.c2f_p4."),
    ("model.16.", "neck.c2f_p3."),
    ("model.17.", "neck.down_p3."),
    ("model.19.", "neck.c2f_n4."),
    ("model.20.", "neck.down_n4."),
    ("model.22.", "neck.c2fcib_n5."),
    ("model.23.", "detect."),
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
        _ensure_yolov10_repo_on_path()
        from ultralytics import YOLOv10

        official = YOLOv10.from_pretrained(model_name).model.float().eval()
        model = cls(conf_threshold=conf_threshold)

        remapped = {}
        for key, value in official.state_dict().items():
            if re.match(r"model\.(11|12|14|15|18|21)\.", key):
                continue
            replaced = key
            for src, dst in _PREFIX_MAP:
                if key.startswith(src):
                    replaced = dst + key[len(src):]
                    break
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
