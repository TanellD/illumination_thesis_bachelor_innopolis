"""
src/models/residual_only.py
===========================
ResidualOnlyModel — noise-residual-only stream with a ResNet-style backbone.

Stage 1 ablation model (experiment B.1).
Input:  [B, 1, 299, 299] noise crop  (or [B, 3, ...] — first 3 channels are
        used for on-the-fly Laplacian extraction if noise_model is None)
Output: [B, num_classes] logits

CRITICAL: the input normalisation uses InstanceNorm2d, NOT BatchNorm2d.
This is intentional — see KNOWN_QUIRKS.md #1.  InstanceNorm is what produced
the reported results; swapping it would change the experiment.

Ported verbatim from scripts/models/ResidualOnly.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.resblock import ResBlock


class ResidualOnlyModel(nn.Module):
    """Noise-residual-only classification head.

    If noise_model is None the model falls back to a Laplacian high-pass
    filter applied to the RGB input.  In reported experiments noise_model
    was a frozen NoiseprintPlusPlus (TruForNoiseModel) instance, but the
    precomputed-crop path (loading from .pt files) bypasses this entirely —
    the dataset returns the crop directly as the noise input.

    The single learnable normalisation is InstanceNorm2d(affine=True) per
    KNOWN_QUIRKS.md #1.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        noise_model=None,
    ):
        super().__init__()
        self.noise_model = noise_model

        # InstanceNorm intentionally kept (see KNOWN_QUIRKS.md #1)
        self.noise_norm = nn.InstanceNorm2d(1, affine=True)

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32,  64,  stride=2),
            ResBlock(64,  128, stride=2),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 512, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _extract_noise(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, :3] if x.shape[1] >= 3 else x
        if self.noise_model is not None:
            return self.noise_model.extract_batch(rgb)
        return _laplacian_noise(rgb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self._extract_noise(x).to(next(self.parameters()).device)
        noise = self.noise_norm(noise)
        feat = self.backbone(noise).flatten(1)
        return self.classifier(feat)


def _laplacian_noise(x: torch.Tensor) -> torch.Tensor:
    """High-pass Laplacian fallback — used when no noise model is provided."""
    k = torch.tensor(
        [[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]],
        dtype=x.dtype, device=x.device,
    ) / 8.0
    channels = [F.conv2d(x[:, c : c + 1], k, padding=1) for c in range(x.shape[1])]
    # channels is a list of [B, 1, H, W] tensors; cat along dim=1 → [B, C, H, W]
    return torch.cat(channels, dim=1).mean(dim=1, keepdim=True)  # → [B, 1, H, W]
