"""Optimized YOLOv10 detection head with fused Triton DFL+dist2bbox kernel and BN fusion."""

from __future__ import annotations

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L2.yolov10_conv import YOLOConv
from fastkernels.tasks.baseline.L2.yolov10_dfl import YOLODFL


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def v10postprocess(preds, max_det, nc=80):
    boxes, scores = preds.split([4, nc], dim=-1)
    max_scores = scores.amax(dim=-1)
    max_scores, index = torch.topk(max_scores, max_det, dim=-1)
    index = index.unsqueeze(-1)
    boxes = torch.gather(boxes, dim=1, index=index.expand(-1, -1, boxes.shape[-1]))
    scores = torch.gather(scores, dim=1, index=index.expand(-1, -1, scores.shape[-1]))
    scores, index = torch.topk(scores.flatten(1), max_det, dim=-1)
    labels = index % nc
    index = index // nc
    boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).expand(-1, -1, boxes.shape[-1]))
    return boxes, scores, labels


def xywh2xyxy(boxes):
    x, y, w, h = boxes.unbind(-1)
    half_w = w * 0.5
    half_h = h * 0.5
    return torch.stack((x - half_w, y - half_h, x + half_w, y + half_h), dim=-1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_A': 64}, num_warps=2),
        triton.Config({'BLOCK_A': 128}, num_warps=4),
        triton.Config({'BLOCK_A': 256}, num_warps=4),
        triton.Config({'BLOCK_A': 512}, num_warps=8),
    ],
    key=['A'],
)
@triton.jit
def _fused_dfl_bbox_kernel(
    box_ptr, anchors_ptr, strides_ptr, out_ptr,
    A: tl.int32,
    stride_box_b: tl.int32,
    stride_out_b: tl.int32,
    BLOCK_A: tl.constexpr,
    REG_MAX: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_a = tl.program_id(1)

    a_start = pid_a * BLOCK_A
    a_idx = a_start + tl.arange(0, BLOCK_A)
    mask = a_idx < A

    anc_x = tl.load(anchors_ptr + a_idx, mask=mask, other=0.0).to(tl.float32)
    anc_y = tl.load(anchors_ptr + A + a_idx, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(strides_ptr + a_idx, mask=mask, other=0.0).to(tl.float32)

    box_base = pid_b * stride_box_b

    # Coord 0 (lt_x): channels [0, REG_MAX)
    m = tl.full([BLOCK_A], float('-inf'), dtype=tl.float32)
    for r in range(REG_MAX):
        v = tl.load(box_ptr + box_base + r * A + a_idx, mask=mask, other=float('-inf')).to(tl.float32)
        m = tl.maximum(m, v)
    se = tl.zeros([BLOCK_A], dtype=tl.float32)
    ws = tl.zeros([BLOCK_A], dtype=tl.float32)
    for r in range(REG_MAX):
        v = tl.load(box_ptr + box_base + r * A + a_idx, mask=mask, other=float('-inf')).to(tl.float32)
        e = tl.exp(v - m)
        se += e
        ws += r * e
    lt_x = ws / se

    # Coord 1 (lt_y): channels [REG_MAX, 2*REG_MAX)
    off1 = REG_MAX * A
    m = tl.full([BLOCK_A], float('-inf'), dtype=tl.float32)
    for r in range(REG_MAX):
        v = tl.load(box_ptr + box_base + off1 + r * A + a_idx, mask=mask, other=float('-inf')).to(tl.float32)
        m = tl.maximum(m, v)
    se = tl.zeros([BLOCK_A], dtype=tl.float32)
    ws = tl.zeros([BLOCK_A], dtype=tl.float32)
    for r in range(REG_MAX):
        v = tl.load(box_ptr + box_base + off1 + r * A + a_idx, mask=mask, other=float('-inf')).to(tl.float32)
        e = tl.exp(v - m)
        se += e
        ws += r * e
    lt_y = ws / se

    # Coord 2 (rb_x): channels [2*REG_MAX, 3*REG_MAX)
    off2 = 2 * REG_MAX * A
    m = tl.full([BLOCK_A], float('-inf'), dtype=tl.float32)
    for r in range(REG_MAX):
        v = tl.load(box_ptr + box_base + off2 + r * A + a_idx, mask=mask, other=float('-inf')).to(tl.float32)
        m = tl.maximum(m, v)
    se = tl.zeros([BLOCK_A], dtype=tl.float32)
    ws = tl.zeros([BLOCK_A], dtype=tl.float32)
    for r in range(REG_MAX):
        v = tl.load(box_ptr + box_base + off2 + r * A + a_idx, mask=mask, other=float('-inf')).to(tl.float32)
        e = tl.exp(v - m)
        se += e
        ws += r * e
    rb_x = ws / se

    # Coord 3 (rb_y): channels [3*REG_MAX, 4*REG_MAX)
    off3 = 3 * REG_MAX * A
    m = tl.full([BLOCK_A], float('-inf'), dtype=tl.float32)
    for r in range(REG_MAX):
        v = tl.load(box_ptr + box_base + off3 + r * A + a_idx, mask=mask, other=float('-inf')).to(tl.float32)
        m = tl.maximum(m, v)
    se = tl.zeros([BLOCK_A], dtype=tl.float32)
    ws = tl.zeros([BLOCK_A], dtype=tl.float32)
    for r in range(REG_MAX):
        v = tl.load(box_ptr + box_base + off3 + r * A + a_idx, mask=mask, other=float('-inf')).to(tl.float32)
        e = tl.exp(v - m)
        se += e
        ws += r * e
    rb_y = ws / se

    # dist2bbox (xywh) * stride
    cx = (anc_x + (rb_x - lt_x) * 0.5) * s
    cy = (anc_y + (rb_y - lt_y) * 0.5) * s
    w = (lt_x + rb_x) * s
    h = (lt_y + rb_y) * s

    out_base = pid_b * stride_out_b
    tl.store(out_ptr + out_base + a_idx, cx, mask=mask)
    tl.store(out_ptr + out_base + A + a_idx, cy, mask=mask)
    tl.store(out_ptr + out_base + 2 * A + a_idx, w, mask=mask)
    tl.store(out_ptr + out_base + 3 * A + a_idx, h, mask=mask)


def _fused_dfl_dist2bbox(box, anchors, strides, reg_max):
    B, C, A = box.shape
    box = box.contiguous()
    out = torch.empty(B, 4, A, device=box.device, dtype=torch.float32)
    grid = lambda meta: (B, triton.cdiv(A, meta['BLOCK_A']))
    _fused_dfl_bbox_kernel[grid](
        box, anchors, strides, out,
        A, C * A, 4 * A,
        REG_MAX=reg_max,
    )
    return out


def _fuse_bn_recursive(module):
    for child in module.children():
        _fuse_bn_recursive(child)
    if isinstance(module, YOLOConv) and not module._is_fused:
        module.fuse()


class YOLOv10DetectHead(nn.Module):
    dynamic = False
    export = True
    shape = None
    max_det = 300

    def __init__(self, nc: int = 80, ch: tuple[int, int, int] = (256, 512, 1024)):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                YOLOConv(x, c2, 3),
                YOLOConv(c2, c2, 3),
                Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(YOLOConv(x, x, 3, g=x), YOLOConv(x, c3, 1)),
                nn.Sequential(YOLOConv(c3, c3, 3, g=c3), YOLOConv(c3, c3, 1)),
                Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = YOLODFL(self.reg_max)
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
        self.register_buffer("anchors", torch.empty(0))
        self.register_buffer("strides", torch.empty(0))
        self._bn_fused = False

    def _ensure_fused(self):
        if not self._bn_fused and not self.training:
            _fuse_bn_recursive(self)
            self._bn_fused = True

    def forward_feat(self, x, cv2, cv3):
        y = []
        for i in range(self.nl):
            y.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return y

    def inference(self, x):
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                t.transpose(0, 1).contiguous()
                for t in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if box.is_cuda:
            dbox = _fused_dfl_dist2bbox(box, self.anchors, self.strides, self.reg_max)
            if dbox.dtype != cls.dtype:
                dbox = dbox.to(cls.dtype)
        else:
            dfl_out = self.dfl(box)
            lt, rb = dfl_out.split([2, 2], 1)
            anchors = self.anchors.unsqueeze(0)
            x1y1 = anchors - lt
            x2y2 = anchors + rb
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            dbox = torch.cat((c_xy, wh), 1) * self.strides

        return torch.cat((dbox, torch.sigmoid(cls)), 1)

    def forward(self, x: list[torch.Tensor]):
        self._ensure_fused()
        one2one = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        if not self.training:
            one2one = self.inference(one2one)
            if self.export:
                boxes, scores, labels = v10postprocess(one2one.permute(0, 2, 1), self.max_det, self.nc)
                return torch.cat([xywh2xyxy(boxes), scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)

        one2many = self.forward_feat(x, self.cv2, self.cv3)
        if self.training:
            return {"one2many": one2many, "one2one": one2one}
        one2many = self.inference(one2many)
        return {"one2many": one2many, "one2one": one2one}

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
        for a, b, s in zip(self.one2one_cv2, self.one2one_cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
