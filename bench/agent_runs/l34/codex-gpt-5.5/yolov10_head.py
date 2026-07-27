from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

try:
    from fastkernels.tasks.baseline.L1.conv2d import Conv2d
    from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
    from fastkernels.tasks.baseline.L2.yolov10_conv import YOLOConv
    from fastkernels.tasks.baseline.L2.yolov10_dfl import YOLODFL
except Exception:
    def _autopad(k, p=None, d=1):
        if p is None:
            if isinstance(k, int):
                p = ((k - 1) * d) // 2
            else:
                p = [((x - 1) * d) // 2 for x in k]
        return p

    class Conv2d(nn.Conv2d):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, bias=True):
            super().__init__(c1, c2, k, s, _autopad(k, p, d), groups=g, dilation=d, bias=bias)

    class Sigmoid(nn.Module):
        def forward(self, x):
            return torch.sigmoid(x)

    class YOLOConv(nn.Module):
        default_act = nn.SiLU()

        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            super().__init__()
            self.conv = Conv2d(c1, c2, k, s, p, g, d, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class YOLODFL(nn.Module):
        def __init__(self, c1=16):
            super().__init__()
            self.c1 = c1
            self.conv = nn.Conv2d(c1, 1, 1, bias=False)
            self.conv.weight.data[:] = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
            self.conv.requires_grad_(False)

        def forward(self, x):
            b, _, a = x.shape
            return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def _make_anchors(feats: list[torch.Tensor], strides: torch.Tensor, grid_cell_offset: float = 0.5):
    anchor_points = []
    stride_tensor = []
    dtype = feats[0].dtype
    device = feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(w, device=device, dtype=dtype).add_(grid_cell_offset)
        sy = torch.arange(h, device=device, dtype=dtype).add_(grid_cell_offset)
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), float(stride), dtype=dtype, device=device))
    return torch.cat(anchor_points, 0), torch.cat(stride_tensor, 0)


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
        self._sigmoid = Sigmoid()
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
        self.register_buffer("anchors", torch.empty(0))
        self.register_buffer("strides", torch.empty(0))
        self.register_buffer("_dfl_proj", torch.arange(self.reg_max, dtype=torch.float32).view(1, 1, self.reg_max, 1), persistent=False)

    def _ensure_anchors(self, x: list[torch.Tensor]):
        shape = x[0].shape
        if self.dynamic or self.shape != shape or self.anchors.device != x[0].device or self.anchors.dtype != x[0].dtype:
            anchors, strides = _make_anchors(x, self.stride, 0.5)
            self.anchors = anchors.transpose(0, 1).contiguous()
            self.strides = strides.transpose(0, 1).contiguous()
            self.shape = shape

    def forward_feat(self, x: list[torch.Tensor], cv2, cv3):
        y = []
        for i in range(self.nl):
            y.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return y

    def _dfl_decode(self, box: torch.Tensor):
        b, _, n = box.shape
        conv = getattr(self.dfl, "conv", None)
        weight = getattr(conv, "weight", None)
        if weight is not None and weight.numel() == self.reg_max:
            proj = weight.reshape(1, 1, self.reg_max, 1).to(device=box.device, dtype=box.dtype)
        else:
            proj = self._dfl_proj.to(device=box.device, dtype=box.dtype)
        return (box.view(b, 4, self.reg_max, n).softmax(2) * proj).sum(2)

    def inference(self, x: list[torch.Tensor]):
        shape = x[0].shape
        b = shape[0]
        x_cat = torch.cat([xi.view(b, self.no, -1) for xi in x], 2)
        self._ensure_anchors(x)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dist = self._dfl_decode(box)
        anchors = self.anchors.unsqueeze(0)
        stride = self.strides.view(1, -1)
        lt, rb = dist.split((2, 2), 1)
        x1y1 = anchors - lt
        x2y2 = anchors + rb
        dbox = torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), 1) * stride
        return torch.cat((dbox, torch.sigmoid(cls)), 1)

    def _forward_export(self, x: list[torch.Tensor]):
        feats = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        self._ensure_anchors(feats)
        b = feats[0].shape[0]
        cls = torch.cat([xi[:, self.reg_max * 4 :, :, :].view(b, self.nc, -1) for xi in feats], 2)
        k = self.max_det
        anchor_index = cls.amax(1).topk(k, dim=1).indices
        selected_cls = torch.gather(cls, 2, anchor_index[:, None, :].expand(-1, self.nc, -1))
        flat_scores, flat_index = selected_cls.permute(0, 2, 1).reshape(b, -1).topk(k, dim=1)
        labels = flat_index.remainder(self.nc)
        local_index = torch.div(flat_index, self.nc, rounding_mode="floor")
        final_anchor_index = torch.gather(anchor_index, 1, local_index)

        box = torch.cat([xi[:, : self.reg_max * 4, :, :].view(b, self.reg_max * 4, -1) for xi in feats], 2)
        selected_box = torch.gather(box, 2, final_anchor_index[:, None, :].expand(-1, self.reg_max * 4, -1))
        dist = self._dfl_decode(selected_box)

        anchor_x = torch.gather(self.anchors[0].view(1, -1).expand(b, -1), 1, final_anchor_index)
        anchor_y = torch.gather(self.anchors[1].view(1, -1).expand(b, -1), 1, final_anchor_index)
        stride = torch.gather(self.strides.view(1, -1).expand(b, -1), 1, final_anchor_index)
        anchors = torch.stack((anchor_x, anchor_y), 1)
        lt, rb = dist.split((2, 2), 1)
        x1y1 = anchors - lt
        x2y2 = anchors + rb
        xywh = torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), 1) * stride[:, None, :]
        cx, cy, width, height = xywh.unbind(1)
        boxes = torch.stack((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), 2)
        scores = torch.sigmoid(flat_scores).unsqueeze(2)
        return torch.cat((boxes, scores, labels.unsqueeze(2).to(boxes.dtype)), 2)

    def forward(self, x: list[torch.Tensor]):
        if not self.training and self.export:
            return self._forward_export(x)

        one2one = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        if not self.training:
            one2one = self.inference(one2one)
            one2many = self.inference(self.forward_feat(x, self.cv2, self.cv3))
            return {"one2many": one2many, "one2one": one2one}

        one2many = self.forward_feat(x, self.cv2, self.cv3)
        return {"one2many": one2many, "one2one": one2one}

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
        for a, b, s in zip(self.one2one_cv2, self.one2one_cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
