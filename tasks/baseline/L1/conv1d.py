"""Conv1d wrapper around nn.Conv1d.

Originally a Whisper-only narrow wrapper (in_channels, out_channels, kernel_size,
stride, padding, bias). Extended additively to accept the full nn.Conv1d
kwarg surface — ``groups``, ``dilation``, ``padding_mode`` — that other HF
audio / TTS / sparse models use (granite_speech, vibevoice, dac, encodec,
pe_audio, squeezebert, etc.). Defaults match torch.nn.Conv1d so existing
callers (Whisper L4) continue to work unchanged.

Internal layout: ``self.conv = nn.Conv1d(...)`` (nested) is preserved so kb-nano
callers that access ``self.conv.weight`` / ``self.conv.bias`` (e.g. Whisper L4
at ``tasks/baseline/L4/whisper.py:117``) keep working.
"""

from __future__ import annotations

import torch.nn as nn


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups,
            bias=bias, padding_mode=padding_mode,
        )

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    @property
    def stride(self):
        return self.conv.stride

    @property
    def padding(self):
        return self.conv.padding

    @property
    def dilation(self):
        return self.conv.dilation

    @property
    def groups(self):
        return self.conv.groups

    def forward(self, x):
        return self.conv(x)
