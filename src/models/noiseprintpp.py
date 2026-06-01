"""
src/models/noiseprintpp.py
==========================
NoiseprintPlusPlus — 17-layer FCN noise extractor (frozen at inference).
TruForNoiseModel  — thin wrapper that keeps the FCN frozen and provides a
                    batch-extraction API used by ResidualOnlyModel and
                    LateFusionModel.

The extract-then-crop order (run on full frame FIRST, crop afterward) is
enforced by the caller (src/noise/precompute.py). This file only provides
the network and the batch API; it does not decide crop boundaries.

Ported verbatim from scripts/models/noiseprintpp.py.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── network utilities ─────────────────────────────────────────────────────────

def _conv(in_planes: int, out_planes: int, kernelsize: int,
          stride: int = 1, dilation: int = 1,
          bias: bool = False, padding: Optional[int] = None) -> nn.Conv2d:
    if padding is None:
        padding = kernelsize // 2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize,
                     stride=stride, dilation=dilation, padding=padding, bias=bias)


def _conv_init(conv: nn.Conv2d) -> None:
    n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2.0 / n))


def _bn_init(bn: nn.BatchNorm2d, kernelsize: int = 3) -> None:
    n = kernelsize ** 2 * bn.num_features
    bn.weight.data.normal_(0, math.sqrt(2.0 / n))
    bn.bias.data.zero_()


def _make_net(nplanes_in: int, kernels: list, features: list,
              bns: list, acts: list, dilats: list,
              bn_momentum: float = 0.1, padding: Optional[int] = None) -> nn.Sequential:
    depth = len(features)
    assert len(features) == len(kernels)
    layers: list = []
    for i in range(depth):
        in_feats = nplanes_in if i == 0 else features[i - 1]
        conv = _conv(in_feats, features[i], kernelsize=kernels[i],
                     dilation=dilats[i], padding=padding,
                     bias=not bns[i])
        _conv_init(conv)
        layers.append(conv)
        if bns[i]:
            bn = nn.BatchNorm2d(features[i], momentum=bn_momentum)
            _bn_init(bn, kernelsize=kernels[i])
            layers.append(bn)
        if acts[i] == "relu":
            layers.append(nn.ReLU(inplace=True))
        # "linear" → no activation appended
    return nn.Sequential(*layers)


# ── NoiseprintPlusPlus ────────────────────────────────────────────────────────

class NoiseprintPlusPlus(nn.Module):
    """17-layer FCN that extracts a single-channel noise residual from an RGB
    image.  Input: [B, 3, H, W]; Output: [B, 1, H, W].

    Weights are expected to be pre-trained and are kept frozen at inference
    time (see TruForNoiseModel.extract_batch).
    """

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__()
        num_levels = 17
        self.net = _make_net(
            3,
            kernels=[3] * num_levels,
            features=[64] * (num_levels - 1) + [1],
            bns=[False] + [True] * (num_levels - 2) + [False],
            acts=["relu"] * (num_levels - 1) + ["linear"],
            dilats=[1] * num_levels,
            bn_momentum=0.1,
            padding=1,
        )
        if weights_path is not None:
            self.load_weights(weights_path)

    def load_weights(self, weights_path: str) -> None:
        state_dict = torch.load(weights_path, map_location="cpu")
        self.net.load_state_dict(state_dict)
        logger.info(f"Loaded NoiseprintPlusPlus weights from {weights_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, 1, H, W]


# ── TruForNoiseModel ──────────────────────────────────────────────────────────

class TruForNoiseModel:
    """Thin wrapper around NoiseprintPlusPlus.

    Kept intentionally frozen (no gradient flow through TruFor weights).
    The downstream noise backbone receives the frozen maps and has a clean
    gradient path through its own parameters.
    """

    def __init__(self, weights_path: Optional[str] = None,
                 device: Optional[torch.device] = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load(weights_path)
        self.model.eval()

    def _load(self, weights_path: Optional[str]) -> nn.Module:
        try:
            if weights_path and os.path.exists(weights_path):
                model = NoiseprintPlusPlus(weights_path)
            else:
                model = NoiseprintPlusPlus()
                logger.warning("TruForNoiseModel: no weights path — using random init")
        except Exception as exc:
            logger.warning(f"TruForNoiseModel load error: {exc} — returning random-init model")
            model = NoiseprintPlusPlus()
        return model.to(self.device)

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass on a batch of RGB images.

        Args:
            images: [B, 3, H, W] on any device
        Returns:
            noise:  [B, 1, H, W] on self.device, float32
        """
        images = images.to(self.device, dtype=torch.float32)
        noise = self.model(images)
        if noise.dim() == 3:
            noise = noise.unsqueeze(1)
        return noise.float()

    def extract_batch_with_cache(
        self,
        images: torch.Tensor,
        cache_dict: Optional[dict] = None,
        image_paths: Optional[list] = None,
    ) -> torch.Tensor:
        """Extract noise with an optional path-keyed in-memory cache."""
        if cache_dict is None or image_paths is None:
            return self.extract_batch(images)

        noise_list = [None] * len(image_paths)
        uncached_idx: list = []
        uncached_images: list = []

        for i, path in enumerate(image_paths):
            if path in cache_dict:
                noise_list[i] = cache_dict[path]
            else:
                uncached_idx.append(i)
                uncached_images.append(images[i : i + 1])

        if uncached_images:
            batch = torch.cat(uncached_images, dim=0)
            noises = self.extract_batch(batch)
            for local_i, global_i in enumerate(uncached_idx):
                n = noises[local_i]
                cache_dict[image_paths[global_i]] = n
                noise_list[global_i] = n

        return torch.stack(noise_list, dim=0)  # type: ignore[arg-type]
