"""
src/robustness/perturbations.py
================================
Pure image→image perturbation functions used by the robustness grid (E.*).

Every function takes a normalised torch tensor batch [B, 3, H, W] in the
[-1, 1] range (ImageNet or Stage 1 normalisation) and returns the perturbed
batch in the same normalisation.  The functions are stateless; the caller
decides when to apply them.

Perturbation families and parameter grids (CLAUDE.md §E):
  JPEG:      quality  ∈ {95, 75, 55, 40}
  Blur:      sigma    ∈ {0.5, 1.0, 2.0, 3.0}
  Ds/sharp:  tags     ∈ {none, denoise, sharpen, both}
  Gamma:     gamma    ∈ {0.5, 0.75, 1.0, 1.5, 2.0}
  Resize:    factor   ∈ {0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0}

Do NOT change these grids — they are the exact conditions reported in the thesis.
Identity conditions (JPEG 95 is NOT identity; gamma 1.0, resize 1.0 are) are
kept in the grid for completeness.

Ported verbatim from scripts/stage1_noise_channel/ablation.py and
scripts/FailureTaxonomy/main_tables_one_script.py.
"""
from __future__ import annotations

import io
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms

# ── canonical grids ───────────────────────────────────────────────────────────

JPEG_QUALITIES       = [95, 75, 55, 40]
BLUR_SIGMAS          = [0.5, 1.0, 2.0, 3.0]
DENOISE_SHARPEN_TAGS = ["none", "denoise", "sharpen", "both"]
GAMMA_VALUES         = [0.5, 0.75, 1.0, 1.5, 2.0]
RESIZE_FACTORS       = [0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0]

# All (family, param_value) pairs in the thesis robustness grid
PERTURBATION_GRID = (
    [("jpeg",   q)   for q   in JPEG_QUALITIES]
    + [("blur",   s)   for s   in BLUR_SIGMAS]
    + [("dssharp", t)  for t   in DENOISE_SHARPEN_TAGS]
    + [("gamma",  g)   for g   in GAMMA_VALUES]
    + [("resize", f)   for f   in RESIZE_FACTORS]
)

# ── normalisation helpers ─────────────────────────────────────────────────────
# Both Stage 1 (mean/std = 0.5) and Stage 2 (ImageNet) pipelines use [-1, 1]
# internally after normalisation.  These helpers work for either convention
# because they only need to know "map to [0,1]" and back.

def _to_01(t: torch.Tensor) -> torch.Tensor:
    """Map a normalised [-1, 1] tensor to [0, 1]."""
    return torch.clamp(t * 0.5 + 0.5, 0.0, 1.0)


def _from_01(t: torch.Tensor) -> torch.Tensor:
    """Map a [0, 1] tensor back to [-1, 1]."""
    return (t - 0.5) / 0.5


# ── perturbation functions ────────────────────────────────────────────────────

def perturb_jpeg(batch: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG re-compress every image in the batch at the given quality."""
    to_pil   = transforms.ToPILImage()
    to_t     = transforms.ToTensor()
    imgs_01  = _to_01(batch).cpu()
    out = []
    for i in range(imgs_01.shape[0]):
        buf = io.BytesIO()
        to_pil(imgs_01[i]).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        out.append(to_t(Image.open(buf).convert("RGB")))
    return _from_01(torch.stack(out)).to(batch.device)


def perturb_blur(batch: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur via separable conv2d (sigma in pixels)."""
    if sigma <= 0.0:
        return batch
    r  = max(1, int(np.ceil(3.0 * sigma)))
    ks = 2 * r + 1
    ax = torch.arange(-r, r + 1, dtype=torch.float32)
    k1 = torch.exp(-ax ** 2 / (2.0 * sigma ** 2))
    k1 = k1 / k1.sum()
    k2 = torch.outer(k1, k1).view(1, 1, ks, ks).repeat(3, 1, 1, 1)
    k2 = k2.to(batch.device)
    imgs_01  = _to_01(batch)
    blurred  = F.conv2d(imgs_01, k2, padding=r, groups=3)
    return _from_01(blurred)


def perturb_denoise_sharpen(
    batch: torch.Tensor,
    tag: str,
) -> torch.Tensor:
    """PIL-based SMOOTH_MORE and/or SHARPEN.

    tag ∈ {"none", "denoise", "sharpen", "both"}.
    """
    denoise = tag in ("denoise", "both")
    sharpen = tag in ("sharpen", "both")
    if not denoise and not sharpen:
        return batch
    to_pil  = transforms.ToPILImage()
    to_t    = transforms.ToTensor()
    imgs_01 = _to_01(batch).cpu()
    out = []
    for i in range(imgs_01.shape[0]):
        pil = to_pil(imgs_01[i])
        if denoise:
            pil = pil.filter(ImageFilter.SMOOTH_MORE)
        if sharpen:
            pil = pil.filter(ImageFilter.SHARPEN)
        out.append(to_t(pil))
    return _from_01(torch.stack(out)).to(batch.device)


def perturb_gamma(batch: torch.Tensor, gamma: float) -> torch.Tensor:
    """Power-law gamma correction."""
    imgs_01  = _to_01(batch)
    corrected = torch.clamp(imgs_01 ** gamma, 0.0, 1.0)
    return _from_01(corrected)


def perturb_resize(batch: torch.Tensor, factor: float) -> torch.Tensor:
    """Bilinear downscale by factor then upscale back to original size."""
    if abs(factor - 1.0) < 1e-6:
        return batch
    _, _, h, w = batch.shape
    h_mid = max(1, int(h * factor))
    w_mid = max(1, int(w * factor))
    imgs_01  = _to_01(batch)
    small    = F.interpolate(imgs_01, size=(h_mid, w_mid),
                              mode="bilinear", align_corners=False)
    restored = F.interpolate(small,   size=(h, w),
                              mode="bilinear", align_corners=False)
    return _from_01(torch.clamp(restored, 0.0, 1.0))


# ── dispatch ──────────────────────────────────────────────────────────────────

def apply_perturbation(
    batch: torch.Tensor,
    family: str,
    param: Union[int, float, str],
) -> torch.Tensor:
    """Dispatch to the correct perturbation function.

    family: 'jpeg' | 'blur' | 'dssharp' | 'gamma' | 'resize'
    param:  the scalar or string parameter for the family
    """
    if family == "jpeg":
        return perturb_jpeg(batch, int(param))
    if family == "blur":
        return perturb_blur(batch, float(param))
    if family == "dssharp":
        return perturb_denoise_sharpen(batch, str(param))
    if family == "gamma":
        return perturb_gamma(batch, float(param))
    if family == "resize":
        return perturb_resize(batch, float(param))
    raise ValueError(f"Unknown perturbation family: {family!r}")
