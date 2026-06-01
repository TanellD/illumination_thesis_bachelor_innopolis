"""
src/models/resblock.py
======================
ResBlock — shared residual block used by ResidualOnlyModel, LateFusionModel,
StatNoiseFusionModel, and ResAwareFusionModel.

The bug in the original Stage 1 code was that the shortcut (skip connection)
was defined but never *added* in the forward pass.  This version correctly
adds it — and has been verified to match the fixed version in
scripts/stage1_noise_channel/training_models.py (fix #1).

See KNOWN_QUIRKS.md for the original bug description.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Standard pre-activation residual block.  Handles stride and channel
    changes via a 1×1 shortcut convolution.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # residual addition (fix #1)
        return F.relu(out, inplace=True)
