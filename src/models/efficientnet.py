"""
src/models/efficientnet.py
==========================
EfficientNet-B4 backbone for Stage 2 (T-SBI) training.

Architecture: tf_efficientnet_b4 from timm, ImageNet pretrained.
Input:  [B, 3, 380, 380]  (B4 native resolution)
Output: [B, 1]  raw logit (binary: real vs fake)
Loss:   BCEWithLogitsLoss  — sigmoid applied externally for inference

Do NOT change this backbone or the 380×380 input size — these are what
produced the reported Stage 2 results. See KNOWN_QUIRKS.md.

Ported from scripts/T_SBI/train.py build_model().
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
INPUT_SIZE    = 380


class EfficientNetB4Model(nn.Module):
    """EfficientNet-B4 binary classifier.

    Produces a single logit per image.  Apply torch.sigmoid() to get p(real).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for EfficientNet-B4. "
                "Install it with: pip install timm"
            )
        self.backbone = timm.create_model(
            "tf_efficientnet_b4", pretrained=pretrained, num_classes=1
        )
        logger.info(
            f"EfficientNetB4Model: "
            f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,} "
            f"trainable params  (pretrained={pretrained})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)   # [B, 1]


def build_model(pretrained: bool = True) -> EfficientNetB4Model:
    """Convenience factory matching the original train.py API."""
    return EfficientNetB4Model(pretrained=pretrained)
