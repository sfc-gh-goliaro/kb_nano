from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L2.yolov10_c2f import YOLOC2f, YOLOC2fCIB
from fastkernels.tasks.baseline.L2.yolov10_concat import YOLOConcat
from fastkernels.tasks.baseline.L2.yolov10_conv import YOLOConv, fuse_module
from fastkernels.tasks.baseline.L2.yolov10_scdown import YOLOSCDown


def _cs(m, x):
    c = m.conv
    return F.silu(F.conv2d(x, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups))


def _cn(m, x):
    c = m.conv
    return F.conv2d(x, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups)


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
        self._optimized = False

    @torch.no_grad()
    def _optimize(self):
        fuse_module(self)
        self.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self._optimized = True

    def forward(self, feats: dict[str, torch.Tensor]):
        if not self._optimized:
            self._optimize()

        cl = torch.channels_last
        p3_bb = feats["p3_backbone"].contiguous(memory_format=cl)
        p4_bb = feats["p4_backbone"].contiguous(memory_format=cl)
        p5_bb = feats["p5_backbone"].contiguous(memory_format=cl)

        x = F.interpolate(p5_bb, scale_factor=2.0, mode="nearest")
        x = torch.cat((x, p4_bb), 1)
        t = _cs(self.c2f_p4.cv1, x)
        a, b = t.chunk(2, 1)
        bt = self.c2f_p4.m[0]
        c = _cs(bt.cv2, _cs(bt.cv1, b))
        p4 = _cs(self.c2f_p4.cv2, torch.cat((a, b, c), 1))

        x = F.interpolate(p4, scale_factor=2.0, mode="nearest")
        x = torch.cat((x, p3_bb), 1)
        t = _cs(self.c2f_p3.cv1, x)
        a, b = t.chunk(2, 1)
        bt = self.c2f_p3.m[0]
        c = _cs(bt.cv2, _cs(bt.cv1, b))
        p3 = _cs(self.c2f_p3.cv2, torch.cat((a, b, c), 1))

        x = _cs(self.down_p3, p3)
        x = torch.cat((x, p4), 1)
        t = _cs(self.c2f_n4.cv1, x)
        a, b = t.chunk(2, 1)
        bt = self.c2f_n4.m[0]
        c = _cs(bt.cv2, _cs(bt.cv1, b))
        n4 = _cs(self.c2f_n4.cv2, torch.cat((a, b, c), 1))

        x = _cn(self.down_n4.cv2, _cs(self.down_n4.cv1, n4))
        x = torch.cat((x, p5_bb), 1)
        t = _cs(self.c2fcib_n5.cv1, x)
        a, b = t.chunk(2, 1)
        cib = self.c2fcib_n5.m[0]
        s = cib.cv1
        c = _cs(s[0], b)
        c = _cs(s[1], c)
        c = F.silu(_cn(s[2].conv, c))
        c = _cs(s[3], c)
        c = _cs(s[4], c)
        c = b + c
        n5 = _cs(self.c2fcib_n5.cv2, torch.cat((a, b, c), 1))

        return [p3, n4, n5]
