import torch
import torch.nn as nn
import torch.nn.functional as F

from fastkernels.tasks.baseline.L2.yolov10_c2f import YOLOC2f
from fastkernels.tasks.baseline.L2.yolov10_conv import YOLOConv
from fastkernels.tasks.baseline.L2.yolov10_scdown import YOLOSCDown
from fastkernels.tasks.baseline.L2.yolov10_sppf import YOLOSPPF


class _Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = YOLOConv(dim, h, 1, act=False)
        self.proj = YOLOConv(dim, dim, 1, act=False)
        self.pe = YOLOConv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        q_t = q.permute(0, 1, 3, 2).contiguous()
        k_t = k.permute(0, 1, 3, 2).contiguous()
        v_t = v.permute(0, 1, 3, 2).contiguous()
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=self.scale)
        x = out.permute(0, 1, 3, 2).reshape(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)


class _PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv(2 * self.c, c1, 1, 1)
        self.attn = _Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(
            YOLOConv(self.c, self.c * 2, 1, 1),
            YOLOConv(self.c * 2, self.c, 1, 1, act=False),
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


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
        self.psa = _PSA(256, 256)
        self._optimized = False

    @torch.no_grad()
    def _optimize(self):
        if self._optimized:
            return
        torch.backends.cudnn.benchmark = True
        for module in self.modules():
            if isinstance(module, YOLOConv) and not module._is_fused:
                module.fuse()
        self.to(memory_format=torch.channels_last)
        self._optimized = True

    def forward(self, x: torch.Tensor):
        if not self._optimized:
            self._optimize()
        x = x.contiguous(memory_format=torch.channels_last)
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
