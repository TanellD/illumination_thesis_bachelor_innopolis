"""
src/models/statnoise_fusion.py
==============================
StatNoiseFusionModel — Xception RGB + 13 scalar noise statistics → MLP.

Stage 1 fixed-fusion variant B.4a.
Input:  face_rgb  [B, 3, 299, 299]
        noise_crop [B, 1, 299, 299]  precomputed noise crop
Output: [B, num_classes] logits

The noise branch intentionally has NO convolutional layers — the bottleneck
diagnostic (B.2) showed that scalar statistics (AUC ~0.747) outperform a
full ConvNet on noise maps (AUC ~0.55-0.58), meaning the signal is
statistical rather than spatial.

The 13 features are:
    mean_abs, std, energy, max_abs, kurtosis, n_pixels,
    p25, p75, p95, iqr, grad_h, grad_w, spatial_entropy

Ported verbatim from scripts/stage1_noise_channel/statnoise.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.xception import xception

N_NOISE_FEATURES = 13


def extract_noise_stats(noise_tensor: torch.Tensor) -> torch.Tensor:
    """Extract 13 scalar statistics from a (C, H, W) noise crop.

    Returns a 1-D float32 tensor of shape (N_NOISE_FEATURES,).
    Uses float64 internally for numerical stability on kurtosis.
    """
    n = noise_tensor.double()
    abs_n = n.abs()
    flat = n.flatten()

    mean_abs = abs_n.mean()
    std_val = flat.std()
    energy = (n ** 2).mean()
    max_abs = abs_n.max()
    n_pixels = float(n.shape[-1] * n.shape[-2])

    mean_val = flat.mean()
    kurtosis = (
        ((flat - mean_val) ** 4).mean() / (std_val ** 4) - 3.0
        if std_val > 1e-10
        else torch.tensor(0.0, dtype=torch.float64)
    )

    sorted_abs = abs_n.flatten().sort()[0]
    n_el = len(sorted_abs)
    p25 = sorted_abs[int(n_el * 0.25)]
    p75 = sorted_abs[int(n_el * 0.75)]
    p95 = sorted_abs[int(n_el * 0.95)]
    iqr = p75 - p25

    n_2d = n.mean(0) if n.dim() == 3 else n
    grad_h = (n_2d[1:, :] - n_2d[:-1, :]).abs().mean()
    grad_w = (n_2d[:, 1:] - n_2d[:, :-1]).abs().mean()

    hist = torch.histc(abs_n.float().flatten(), bins=32,
                       min=0, max=float(abs_n.max()) + 1e-8)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -(hist * hist.log()).sum()

    return torch.tensor([
        mean_abs, std_val, energy, max_abs,
        kurtosis, n_pixels,
        p25, p75, p95, iqr,
        grad_h, grad_w, entropy,
    ], dtype=torch.float32)


class StatNoiseFusionModel(nn.Module):
    """Xception RGB + scalar noise statistics → MLP fusion."""

    RGB_DIM       = 2048
    NOISE_FEAT_DIM = 32

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super().__init__()

        # ── RGB branch ────────────────────────────────────────────────────
        self.rgb_backbone = xception(
            num_classes=1000, pretrained="imagenet", dropout_rate=dropout_rate
        )
        self.rgb_backbone.last_linear = nn.Identity()
        self._freeze_rgb_early()

        # ── Noise branch (no convolutions) ───────────────────────────────
        self.noise_mlp = nn.Sequential(
            nn.BatchNorm1d(N_NOISE_FEATURES),
            nn.Linear(N_NOISE_FEATURES, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, self.NOISE_FEAT_DIM),
            nn.ReLU(inplace=True),
        )

        # ── Fusion classifier ─────────────────────────────────────────────
        fuse = self.RGB_DIM + self.NOISE_FEAT_DIM
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

    def forward(self, face_rgb: torch.Tensor,
                noise_crop: torch.Tensor) -> torch.Tensor:
        """
        face_rgb:   [B, 3, 299, 299] normalised [-1, 1]
        noise_crop: [B, 1, 299, 299] precomputed noise crop (.pt file content)
        """
        dev = next(self.parameters()).device

        # RGB stream
        rgb_feat = F.adaptive_avg_pool2d(
            self.rgb_backbone.relu(
                self.rgb_backbone.features(face_rgb.to(dev))), 1
        ).flatten(1)   # [B, 2048]

        # Noise stream — scalar stats extracted per sample, cheap
        noise_stats = torch.stack([
            extract_noise_stats(noise_crop[i]) for i in range(noise_crop.shape[0])
        ]).to(dev)   # [B, N_NOISE_FEATURES]
        noise_feat = self.noise_mlp(noise_stats)  # [B, 32]

        return self.classifier(torch.cat([rgb_feat, noise_feat], dim=1))
