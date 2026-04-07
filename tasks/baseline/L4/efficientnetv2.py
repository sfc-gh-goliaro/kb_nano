"""EfficientNetV2 image classification model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d
from ..L1.linear import Linear
from ..L2.efficientnetv2_blocks import BatchNormAct2d
from ..L3.efficientnetv2_stage import EfficientNetV2Stage


@dataclass
class ImageClassifierOutput:
    logits: torch.Tensor


class EfficientNetV2ClassifierHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.flatten = nn.Flatten(1)
        self.classifier = Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.flatten(x))


class EfficientNetV2ForImageClassification(nn.Module):
    def __init__(self, stem_out: int, stage_specs: list[list[dict]], head_out: int, num_classes: int):
        super().__init__()
        self.conv_stem = Conv2d(3, stem_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNormAct2d(stem_out, act_layer=nn.SiLU(inplace=True))
        self.blocks = nn.Sequential(*[EfficientNetV2Stage(specs).blocks for specs in stage_specs])
        final_block_out = stage_specs[-1][-1]["out_chs"]
        self.conv_head = Conv2d(final_block_out, head_out, kernel_size=1, bias=False)
        self.bn2 = BatchNormAct2d(head_out, act_layer=nn.SiLU(inplace=True))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = Linear(head_out, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> ImageClassifierOutput:
        x = self.conv_stem(pixel_values)
        x = self.bn1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.global_pool(x).flatten(1)
        logits = self.classifier(x)
        return ImageClassifierOutput(logits=logits)
