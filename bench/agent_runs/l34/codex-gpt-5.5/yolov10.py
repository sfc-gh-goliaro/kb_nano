from __future__ import annotations

import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from fastkernels.tasks.baseline.L2.yolov10_conv import fuse_module
from fastkernels.tasks.baseline.L3.yolov10_backbone import YOLOv10Backbone
from fastkernels.tasks.baseline.L3.yolov10_head import (
    YOLOv10DetectHead,
    dist2bbox,
    make_anchors,
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


@triton.jit
def _dfl_bbox_kernel(
    box_ptr,
    anchors_ptr,
    strides_ptr,
    out_ptr,
    a_count: tl.constexpr,
    stride_box_b: tl.constexpr,
    stride_out_b: tl.constexpr,
    block_a: tl.constexpr,
    reg_max: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_a = tl.program_id(1)
    offs = pid_a * block_a + tl.arange(0, block_a)
    mask = offs < a_count

    anc_x = tl.load(anchors_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    anc_y = tl.load(anchors_ptr + a_count + offs, mask=mask, other=0.0).to(tl.float32)
    stride = tl.load(strides_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    base = pid_b * stride_box_b

    m0 = tl.full([block_a], -float("inf"), dtype=tl.float32)
    m1 = tl.full([block_a], -float("inf"), dtype=tl.float32)
    m2 = tl.full([block_a], -float("inf"), dtype=tl.float32)
    m3 = tl.full([block_a], -float("inf"), dtype=tl.float32)
    for r in range(reg_max):
        m0 = tl.maximum(m0, tl.load(box_ptr + base + r * a_count + offs, mask=mask, other=-float("inf")).to(tl.float32))
        m1 = tl.maximum(m1, tl.load(box_ptr + base + (reg_max + r) * a_count + offs, mask=mask, other=-float("inf")).to(tl.float32))
        m2 = tl.maximum(m2, tl.load(box_ptr + base + (2 * reg_max + r) * a_count + offs, mask=mask, other=-float("inf")).to(tl.float32))
        m3 = tl.maximum(m3, tl.load(box_ptr + base + (3 * reg_max + r) * a_count + offs, mask=mask, other=-float("inf")).to(tl.float32))

    s0 = tl.zeros([block_a], dtype=tl.float32)
    s1 = tl.zeros([block_a], dtype=tl.float32)
    s2 = tl.zeros([block_a], dtype=tl.float32)
    s3 = tl.zeros([block_a], dtype=tl.float32)
    w0 = tl.zeros([block_a], dtype=tl.float32)
    w1 = tl.zeros([block_a], dtype=tl.float32)
    w2 = tl.zeros([block_a], dtype=tl.float32)
    w3 = tl.zeros([block_a], dtype=tl.float32)
    for r in range(reg_max):
        v0 = tl.load(box_ptr + base + r * a_count + offs, mask=mask, other=-float("inf")).to(tl.float32)
        v1 = tl.load(box_ptr + base + (reg_max + r) * a_count + offs, mask=mask, other=-float("inf")).to(tl.float32)
        v2 = tl.load(box_ptr + base + (2 * reg_max + r) * a_count + offs, mask=mask, other=-float("inf")).to(tl.float32)
        v3 = tl.load(box_ptr + base + (3 * reg_max + r) * a_count + offs, mask=mask, other=-float("inf")).to(tl.float32)
        e0 = tl.exp(v0 - m0)
        e1 = tl.exp(v1 - m1)
        e2 = tl.exp(v2 - m2)
        e3 = tl.exp(v3 - m3)
        rf = r + 0.0
        s0 += e0
        s1 += e1
        s2 += e2
        s3 += e3
        w0 += rf * e0
        w1 += rf * e1
        w2 += rf * e2
        w3 += rf * e3

    lt_x = w0 / s0
    lt_y = w1 / s1
    rb_x = w2 / s2
    rb_y = w3 / s3
    out_base = pid_b * stride_out_b
    tl.store(out_ptr + out_base + offs, (anc_x + (rb_x - lt_x) * 0.5) * stride, mask=mask)
    tl.store(out_ptr + out_base + a_count + offs, (anc_y + (rb_y - lt_y) * 0.5) * stride, mask=mask)
    tl.store(out_ptr + out_base + 2 * a_count + offs, (lt_x + rb_x) * stride, mask=mask)
    tl.store(out_ptr + out_base + 3 * a_count + offs, (lt_y + rb_y) * stride, mask=mask)


def _fused_dfl_dist2bbox(box: torch.Tensor, anchors: torch.Tensor, strides: torch.Tensor, reg_max: int) -> torch.Tensor:
    b, c, a = box.shape
    box = box.contiguous()
    out = torch.empty((b, 4, a), device=box.device, dtype=torch.float32)
    block_a = 256
    grid = (b, triton.cdiv(a, block_a))
    _dfl_bbox_kernel[grid](
        box,
        anchors.contiguous(),
        strides.contiguous(),
        out,
        a,
        c * a,
        4 * a,
        block_a,
        reg_max,
        num_warps=4,
    )
    return out


def _patched_attn_forward(self, x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    n = h * w
    qkv = self.qkv(x)
    q, k, v = qkv.reshape(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
        [self.key_dim, self.key_dim, self.head_dim],
        dim=2,
    )
    if x.is_cuda:
        q = q.transpose(-2, -1).contiguous()
        k = k.transpose(-2, -1).contiguous()
        v_attn = v.transpose(-2, -1).contiguous()
        y = F.scaled_dot_product_attention(q, k, v_attn, scale=self.scale)
        x = y.transpose(-2, -1).reshape(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
    else:
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = self._softmax(attn)
        x = (v @ attn.transpose(-2, -1)).reshape(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
    return self.proj(x)


def _patched_detect_inference(self, x: list[torch.Tensor]) -> torch.Tensor:
    shape = x[0].shape
    x_cat = torch.cat([xi.reshape(shape[0], self.no, -1) for xi in x], 2)
    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (t.transpose(0, 1).contiguous() for t in make_anchors(x, self.stride, 0.5))
        self.shape = shape
    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    if box.is_cuda:
        dbox = _fused_dfl_dist2bbox(box, self.anchors, self.strides, self.reg_max)
        if dbox.dtype != cls.dtype:
            dbox = dbox.to(cls.dtype)
    else:
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    return torch.cat((dbox, torch.sigmoid(cls)), 1)


class YOLOv10ForObjectDetection(nn.Module):
    def __init__(self, conf_threshold: float = 0.25):
        super().__init__()
        self.backbone = YOLOv10Backbone()
        self.neck = YOLOv10Neck()
        self.detect = YOLOv10DetectHead(nc=80, ch=(64, 128, 256))
        self.conf_threshold = conf_threshold
        self._cuda_graphs = {}
        self._capture_failed = False
        self._optimized = False
        self._use_channels_last = False

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
            for src, dst in _PREFIX_MAP:
                if key.startswith(src):
                    remapped[dst + key[len(src):]] = value
                    break

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
        model._optimize_inference()
        return model

    def eval(self):
        return super().eval()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._optimized = False
        self._cuda_graphs.clear()
        self._capture_failed = False
        if not self.training:
            self._optimize_inference()
        return result

    @torch.no_grad()
    def _optimize_inference(self) -> None:
        if self._optimized:
            return
        try:
            fuse_module(self)
        except Exception:
            pass
        self.detect.export = True
        try:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
        except StopIteration:
            self._optimized = True
            return
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            try:
                self.backbone.psa.attn.forward = types.MethodType(_patched_attn_forward, self.backbone.psa.attn)
            except Exception:
                pass
            try:
                self.detect.inference = types.MethodType(_patched_detect_inference, self.detect)
            except Exception:
                pass
            try:
                self.to(memory_format=torch.channels_last)
                self._use_channels_last = True
            except Exception:
                self._use_channels_last = False
            self._warmup_and_capture(device, dtype)
        self._optimized = True

    def _raw_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values)
        pyramid = self.neck(feats)
        return self.detect(pyramid)

    @torch.no_grad()
    def _warmup_and_capture(self, device, dtype: torch.dtype) -> None:
        if not torch.cuda.is_available():
            return
        try:
            shape = (1, 3, 640, 640)
            static_input = torch.empty(shape, device=device, dtype=dtype).contiguous(memory_format=torch.channels_last)
            static_input.zero_()
            stream = torch.cuda.Stream(device=static_input.device)
            stream.wait_stream(torch.cuda.current_stream(static_input.device))
            with torch.cuda.stream(stream):
                for _ in range(4):
                    self._raw_forward(static_input)
            torch.cuda.current_stream(static_input.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = self._raw_forward(static_input)
            self._cuda_graphs[(shape, dtype, static_input.device.index)] = (graph, static_input, static_output)
        except Exception:
            self._cuda_graphs.clear()
            self._capture_failed = True

    @torch.no_grad()
    def _capture_for_input(self, pixel_values: torch.Tensor) -> None:
        if self._capture_failed or self.training or not pixel_values.is_cuda:
            return
        key = (tuple(pixel_values.shape), pixel_values.dtype, pixel_values.device.index)
        if key in self._cuda_graphs:
            return
        try:
            static_input = torch.empty_strided(
                tuple(pixel_values.shape),
                pixel_values.stride(),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
            static_input.zero_()
            stream = torch.cuda.Stream(device=pixel_values.device)
            stream.wait_stream(torch.cuda.current_stream(pixel_values.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._raw_forward(static_input)
            torch.cuda.current_stream(pixel_values.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = self._raw_forward(static_input)
            self._cuda_graphs[key] = (graph, static_input, static_output)
        except Exception:
            self._cuda_graphs.clear()
            self._capture_failed = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if not self.training and not self._optimized:
            self._optimize_inference()
        if self._use_channels_last and pixel_values.dim() == 4:
            pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
        if not self.training and pixel_values.is_cuda:
            key = (tuple(pixel_values.shape), pixel_values.dtype, pixel_values.device.index)
            entry = self._cuda_graphs.get(key)
            if entry is not None:
                graph, static_input, static_output = entry
                static_input.copy_(pixel_values)
                graph.replay()
                return static_output.clone()
        if self.training:
            return self._raw_forward(pixel_values)
        with torch.inference_mode():
            out = self._raw_forward(pixel_values)
            self._capture_for_input(pixel_values)
            return out

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

        batch, candidates = scores.shape
        padded_boxes = boxes.new_zeros((batch, max_detections, 4))
        padded_scores = scores.new_zeros((batch, max_detections))
        padded_labels = torch.full((batch, max_detections), -1, device=boxes.device, dtype=torch.long)
        if candidates == 0 or max_detections == 0:
            return {"boxes": padded_boxes, "scores": padded_scores, "labels": padded_labels}

        keep = scores > self.conf_threshold
        ranks = keep.to(torch.int32).cumsum(dim=1) - 1
        selected = keep & (ranks < max_detections)
        if selected.any():
            batch_idx = torch.arange(batch, device=boxes.device).view(batch, 1).expand(batch, candidates)
            dst_b = batch_idx[selected]
            dst_i = ranks[selected].long()
            padded_boxes[dst_b, dst_i] = boxes[selected]
            padded_scores[dst_b, dst_i] = scores[selected]
            padded_labels[dst_b, dst_i] = labels[selected]

        return {
            "boxes": padded_boxes,
            "scores": padded_scores,
            "labels": padded_labels,
        }
