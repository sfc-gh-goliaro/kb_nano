from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L2.yolov10_c2f import YOLOC2f, YOLOC2fCIB
from fastkernels.tasks.baseline.L2.yolov10_concat import YOLOConcat
from fastkernels.tasks.baseline.L2.yolov10_conv import YOLOConv, fuse_module
from fastkernels.tasks.baseline.L2.yolov10_scdown import YOLOSCDown


def _conv_silu(m, x):
    c = m.conv
    return F.silu(F.conv2d(x, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups))


def _conv(m, x):
    c = m.conv
    return F.conv2d(x, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups)


if triton is not None:
    @triton.jit
    def _upsample2_cat_kernel(
        a, b, out,
        total: tl.constexpr,
        ca: tl.constexpr,
        cb: tl.constexpr,
        ho: tl.constexpr,
        wo: tl.constexpr,
        sa0: tl.constexpr,
        sa1: tl.constexpr,
        sa2: tl.constexpr,
        sa3: tl.constexpr,
        sb0: tl.constexpr,
        sb1: tl.constexpr,
        sb2: tl.constexpr,
        sb3: tl.constexpr,
        so0: tl.constexpr,
        so1: tl.constexpr,
        so2: tl.constexpr,
        so3: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        csum = ca + cb
        ch = offs % csum
        t = offs // csum
        w = t % wo
        t = t // wo
        h = t % ho
        n = t // ho
        from_a = ch < ca
        a_off = n * sa0 + ch * sa1 + (h // 2) * sa2 + (w // 2) * sa3
        b_off = n * sb0 + (ch - ca) * sb1 + h * sb2 + w * sb3
        v = tl.load(a + a_off, mask & from_a, other=0.0)
        v = tl.where(from_a, v, tl.load(b + b_off, mask & (~from_a), other=0.0))
        tl.store(out + n * so0 + ch * so1 + h * so2 + w * so3, v, mask)


def _upsample2_cat(a, b):
    if triton is None or not a.is_cuda:
        return torch.cat((F.interpolate(a, scale_factor=2.0, mode="nearest"), b), 1)
    n, ca, hi, wi = a.shape
    cb = b.shape[1]
    ho = hi * 2
    wo = wi * 2
    out = torch.empty(
        (n, ca + cb, ho, wo),
        device=a.device,
        dtype=a.dtype,
        memory_format=torch.channels_last,
    )
    total = n * (ca + cb) * ho * wo
    grid = (triton.cdiv(total, 256),)
    _upsample2_cat_kernel[grid](
        a, b, out,
        total, ca, cb, ho, wo,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK=256,
    )
    return out


def _c2f_1(m, x):
    t = _conv_silu(m.cv1, x)
    a, b = t.chunk(2, 1)
    bt = m.m[0]
    c = _conv_silu(bt.cv2, _conv_silu(bt.cv1, b))
    return _conv_silu(m.cv2, torch.cat((a, b, c), 1))


def _c2fcib_1(m, x):
    t = _conv_silu(m.cv1, x)
    a, b = t.chunk(2, 1)
    cib = m.m[0]
    s = cib.cv1
    c = _conv_silu(s[0], b)
    c = _conv_silu(s[1], c)
    c = F.silu(_conv(s[2].conv, c))
    c = _conv_silu(s[3], c)
    c = _conv_silu(s[4], c)
    c = b + c
    return _conv_silu(m.cv2, torch.cat((a, b, c), 1))


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
        if self._optimized:
            return
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
        p3_backbone = feats["p3_backbone"].contiguous(memory_format=cl)
        p4_backbone = feats["p4_backbone"].contiguous(memory_format=cl)
        p5_backbone = feats["p5_backbone"].contiguous(memory_format=cl)

        p4 = _c2f_1(self.c2f_p4, _upsample2_cat(p5_backbone, p4_backbone))
        p3 = _c2f_1(self.c2f_p3, _upsample2_cat(p4, p3_backbone))

        x = torch.cat((_conv_silu(self.down_p3, p3), p4), 1)
        n4 = _c2f_1(self.c2f_n4, x)

        x = torch.cat((_conv(self.down_n4.cv2, _conv_silu(self.down_n4.cv1, n4)), p5_backbone), 1)
        n5 = _c2fcib_1(self.c2fcib_n5, x)
        return [p3, n4, n5]
