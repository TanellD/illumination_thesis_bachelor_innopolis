"""
src/models/rgb_only.py
======================
RGBOnlyModel — Xception backbone, RGB input only.

Stage 1 baseline model (experiment B.1).
Input:  [B, 3, 299, 299]  (4-channel input is silently sliced to [:, :3])
Output: [B, num_classes]  logits

Ported verbatim from scripts/models/RGB_only.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.xception import xception


class RGBOnlyModel(nn.Module):
    """Xception backbone with a 3-layer classification head.

    Early Xception layers (conv1, bn1, conv2, bn2, block1–block3) are frozen
    at construction time.  The rest of the backbone and the head are trainable.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        self.backbone = xception(
            num_classes=1000, pretrained="imagenet", dropout_rate=dropout_rate
        )
        self.backbone.last_linear = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        self._freeze_early_layers()

    def _freeze_early_layers(self) -> None:
        frozen = {"conv1", "bn1", "conv2", "bn2", "block1", "block2", "block3"}
        for name, param in self.backbone.named_parameters():
            if any(f in name for f in frozen):
                param.requires_grad = False

    def unfreeze_blocks(self, blocks: list[str]) -> None:
        """Selectively unfreeze named Xception blocks (used by progressive unfreezing)."""
        for name, param in self.backbone.named_parameters():
            if any(b in name for b in blocks):
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 4:
            x = x[:, :3]
        feat = self.backbone.features(x)
        feat = self.backbone.relu(feat)
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return self.classifier(feat)
