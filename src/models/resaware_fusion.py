"""
src/models/resaware_fusion.py
=============================
ResAwareFusionModel — Xception RGB + resolution-aware noise branch.

Stage 1 fixed-fusion variant B.4b.
Input:  face_rgb   [B, 3, 299, 299]
        noise_crop [B, C, H, W]  — H,W can be 299 (legacy) or variable (native)
Output: [B, num_classes] logits

Key differences from LateFusionModel:
  - InstanceNorm2d instead of BatchNorm2d in noise branch (see KNOWN_QUIRKS.md)
  - AdaptiveAvgPool2d accepts any spatial size → native-resolution noise crops
  - Output feature dim is 512 (vs 256 in Late-Fusion) because of the extra fc

Ported verbatim from scripts/stage1_noise_channel/resolution_aware_model_fixed.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.xception import xception
from src.models.resblock import ResBlock


class ResAwareFusionModel(nn.Module):
    """Xception RGB + resolution-aware noise branch (InstanceNorm, variable size)."""

    RGB_DIM      = 2048
    RESIDUAL_DIM = 512

    def __init__(
        self,
        noise_in_channels: int = 1,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.noise_in_channels = noise_in_channels

        # ── RGB branch ────────────────────────────────────────────────────
        self.rgb_backbone = xception(
            num_classes=1000, pretrained="imagenet", dropout_rate=dropout_rate
        )
        self.rgb_backbone.last_linear = nn.Identity()
        self._freeze_rgb_early()

        # ── Noise branch (resolution-aware) ──────────────────────────────
        self.noise_norm = nn.InstanceNorm2d(noise_in_channels, affine=True)
        self.noise_backbone = nn.Sequential(
            nn.Conv2d(noise_in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResBlock(32,  64,  stride=2),
            ResBlock(64,  128, stride=2),
            ResBlock(128, 256, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),   # accepts any spatial size
        )
        self.noise_fc = nn.Sequential(
            nn.Linear(256, self.RESIDUAL_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self._init_noise_weights()

        # ── Fusion classifier ─────────────────────────────────────────────
        fuse = self.RGB_DIM + self.RESIDUAL_DIM
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fuse, 1024),  nn.ReLU(inplace=True), nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(1024, 512),   nn.ReLU(inplace=True), nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),    nn.ReLU(inplace=True), nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_classes),
        )

    def _freeze_rgb_early(self) -> None:
        frozen = {"conv1", "bn1", "block1", "block2", "block3"}
        for name, param in self.rgb_backbone.named_parameters():
            param.requires_grad = not any(f in name for f in frozen)

    def unfreeze_rgb_blocks(self, blocks: list[str]) -> None:
        for name, param in self.rgb_backbone.named_parameters():
            if any(b in name for b in blocks):
                param.requires_grad = True

    def _init_noise_weights(self) -> None:
        for m in self.noise_backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, face_rgb: torch.Tensor,
                noise_crop: torch.Tensor) -> torch.Tensor:
        """
        face_rgb:   [B, 3, 299, 299] normalised [-1, 1]
        noise_crop: [B, C, H, W]  — variable H,W supported
        """
        dev = next(self.parameters()).device

        # RGB stream
        rgb_feat = F.adaptive_avg_pool2d(
            self.rgb_backbone.relu(
                self.rgb_backbone.features(face_rgb.to(dev))), 1
        ).flatten(1)   # [B, 2048]

        # Noise stream
        noise_feat = self.noise_backbone(
            self.noise_norm(noise_crop.to(dev))
        ).flatten(1)   # [B, 256]
        noise_feat = self.noise_fc(noise_feat)  # [B, 512]

        return self.classifier(torch.cat([rgb_feat, noise_feat], dim=1))
