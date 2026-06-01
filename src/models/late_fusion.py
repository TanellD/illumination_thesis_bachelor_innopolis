"""
src/models/late_fusion.py
=========================
LateFusionModel — pretrained Xception (RGB) + lightweight ResNet (noise),
concatenated and classified jointly.

Stage 1 fusion model (experiment B.1).
Input:  [B, 3, 299, 299]  RGB crop  AND  [B, 1, 299, 299] noise crop
        (if only one tensor is passed the model extracts noise on the fly)
Output: [B, num_classes] logits

CRITICAL: the noise normalisation uses BatchNorm2d, NOT InstanceNorm2d.
See KNOWN_QUIRKS.md #2 — do NOT change this.

Ported verbatim from scripts/models/LateFusion.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.xception import xception
from src.models.resblock import ResBlock
from src.models.residual_only import _laplacian_noise


class LateFusionModel(nn.Module):
    """RGB stream (Xception) + noise stream (lightweight ResNet), late fusion."""

    RGB_DIM      = 2048
    RESIDUAL_DIM = 256

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        noise_model=None,
    ):
        super().__init__()
        self.noise_model = noise_model

        # ── RGB stream (Xception) ─────────────────────────────────────────
        self.rgb_backbone = xception(
            num_classes=1000, pretrained="imagenet", dropout_rate=dropout_rate
        )
        self.rgb_backbone.last_linear = nn.Identity()
        self._freeze_rgb_early()

        # ── Noise stream (lightweight ResNet) ────────────────────────────
        # BatchNorm2d intentionally kept here — see KNOWN_QUIRKS.md #2
        self.noise_norm = nn.BatchNorm2d(1, affine=True)
        self.noise_backbone = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResBlock(32,  64,  stride=2),
            ResBlock(64,  128, stride=2),
            ResBlock(128, self.RESIDUAL_DIM, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
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

    def _extract_noise(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, :3] if x.shape[1] >= 3 else x
        if self.noise_model is not None:
            return self.noise_model.extract_batch(rgb)
        return _laplacian_noise(rgb)

    def forward(self, x: torch.Tensor,
                noise: torch.Tensor | None = None) -> torch.Tensor:
        """
        x     : [B, 3, 299, 299] RGB crop
        noise : [B, 1, 299, 299] precomputed noise crop (optional).
                If None, extracted on the fly from x.
        """
        dev = next(self.parameters()).device
        rgb_in = x[:, :3] if x.shape[1] >= 3 else x

        # RGB stream
        rgb_feat = self.rgb_backbone.features(rgb_in)
        rgb_feat = self.rgb_backbone.relu(rgb_feat)
        rgb_feat = F.adaptive_avg_pool2d(rgb_feat, (1, 1)).flatten(1)

        # Noise stream
        if noise is None:
            noise = self._extract_noise(x)
        noise = noise.to(dev)
        noise_feat = self.noise_backbone(self.noise_norm(noise)).flatten(1)

        return self.classifier(torch.cat([rgb_feat, noise_feat], dim=1))
