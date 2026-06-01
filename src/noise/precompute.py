"""
Noiseprint++ noise-map precomputation.

CRITICAL ORDER: extract noise on the FULL frame, then crop to the face bbox.
Running Noiseprint++ on the face crop first gives wrong results because the
FCN exploits spatial correlations across the entire image.  See KNOWN_QUIRKS #3.

For each row in the input manifest this module:
  1. Loads the full source image (full_frame_path column if present,
     otherwise falls back to face_crop_path for datasets without full frames).
  2. Runs NoiseprintPlusPlus on the full image → [1, H, W] noise map.
  3. Crops the noise map to the face bbox and resizes to crop_size × crop_size.
  4. Saves both the cropped noise ([1, crop_size, crop_size]) and the full-frame
     noise ([1, H, W]) as .pt files with content-addressed names.
  5. Writes an updated manifest CSV with noise_crop_path and noise_full_path.

Idempotent: skips rows whose .pt files already exist (unless --force).
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.data.manifest import add_noise_paths

logger = logging.getLogger(__name__)

_to_tensor = transforms.ToTensor()   # converts PIL → [0, 1] float32 CHW


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

class _LaplacianFallback(torch.nn.Module):
    """Laplacian high-pass filter used when NoiseprintPlusPlus weights are absent."""

    def __init__(self) -> None:
        super().__init__()
        k = torch.tensor([[[[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]]]], dtype=torch.float32) / 8.0
        self.register_buffer("k", k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]  →  mean of per-channel Laplacian, shape [B, 1, H, W]
        return torch.stack(
            [F.conv2d(x[:, c:c+1], self.k, padding=1) for c in range(x.shape[1])],
            dim=1,
        ).mean(dim=1, keepdim=True)


def load_noise_model(weights_path: Optional[str], device: torch.device):
    """Load NoiseprintPlusPlus, falling back to Laplacian if weights are absent."""
    try:
        from src.models.noiseprintpp import NoiseprintPlusPlus
        if weights_path and os.path.exists(weights_path):
            model = NoiseprintPlusPlus(weights_path)
        else:
            model = NoiseprintPlusPlus()
        logger.info("NoiseprintPlusPlus loaded on %s", device)
    except Exception as exc:
        logger.warning(
            "NoiseprintPlusPlus unavailable (%s) — using Laplacian fallback", exc
        )
        model = _LaplacianFallback()

    return model.to(device).eval()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _stable_stem(path: str) -> str:
    """Content-addressed filename: SHA-256[:16] of the path string."""
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


def _load_pil(path: str, fallback: tuple = (299, 299)) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", fallback, 0)


@torch.no_grad()
def _run_noiseprint(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
) -> torch.Tensor:
    """Run the noise model on the image at image_path.

    Returns: [1, H, W] float32 CPU tensor (full-frame noise map).

    IMPORTANT: This runs on the FULL frame, not the face crop.  The crop step
    happens afterward in crop_noise_to_face().  Do not reorder. (KNOWN_QUIRKS #3)
    """
    pil    = _load_pil(image_path)
    tensor = _to_tensor(pil).unsqueeze(0).to(device, dtype=torch.float32)  # [1,3,H,W]
    noise  = model(tensor)                                                  # [1,1,H,W]
    result = noise.squeeze(0).cpu()                                         # [1,H,W]
    del tensor, noise
    return result


def crop_noise_to_face(
    noise_full: torch.Tensor,
    bbox: tuple,
    out_size: int = 299,
) -> torch.Tensor:
    """Crop the full-frame noise map to the face bbox and resize.

    Args:
        noise_full: [1, H, W] CPU tensor.
        bbox:       (x1, y1, x2, y2) integer pixel coordinates.
        out_size:   Output spatial size (default 299 for Stage 1).

    Returns: [1, out_size, out_size] float32 CPU tensor.
    """
    noise4d = noise_full.unsqueeze(0)          # [1, 1, H, W]
    H       = noise4d.shape[2]
    W       = noise4d.shape[3]

    x1 = max(0,     min(int(bbox[0]), W - 1))
    y1 = max(0,     min(int(bbox[1]), H - 1))
    x2 = max(x1+1,  min(int(bbox[2]), W))
    y2 = max(y1+1,  min(int(bbox[3]), H))

    cropped = F.interpolate(
        noise4d[:, :, y1:y2, x1:x2],
        size=(out_size, out_size),
        mode="bilinear",
        align_corners=False,
    )                                          # [1, 1, out_size, out_size]
    return cropped.squeeze(0)                  # [1, out_size, out_size]


# ──────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ──────────────────────────────────────────────────────────────────────────────

def run_precompute(
    manifest_csv: str,
    output_dir: str,
    weights_path: Optional[str] = None,
    crop_size: int = 299,
    split_filter: Optional[str] = None,
    force: bool = False,
    gc_interval: int = 200,
) -> str:
    """Precompute noise maps for every row in manifest_csv.

    Args:
        manifest_csv:  Path to the face-crop manifest CSV.
        output_dir:    Root directory for .pt files and the updated manifest.
        weights_path:  Path to NoiseprintPlusPlus .pth weights.
        crop_size:     Spatial size of the cropped noise map (299 for Stage 1).
        split_filter:  If given, process only rows with manifest['split'] == split_filter.
        force:         Re-extract even if .pt files already exist.
        gc_interval:   Run torch.cuda.empty_cache() every N rows.

    Returns: Path to the updated manifest CSV with noise_crop_path / noise_full_path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_noise_model(weights_path, device)

    df = pd.read_csv(manifest_csv)
    if split_filter:
        df = df[df["split"] == split_filter].reset_index(drop=True)

    has_bbox = all(c in df.columns for c in ("x1", "y1", "x2", "y2"))

    # Choose the full-frame path column if available; fall back to face_crop_path.
    # Datasets that save full frames (DFDC) have a full_frame_path column.
    use_full_frame = "full_frame_path" in df.columns

    crops_dir = os.path.join(output_dir, "noise_crops")
    fulls_dir = os.path.join(output_dir, "noise_fulls")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(fulls_dir, exist_ok=True)

    noise_crop_paths: list = []
    noise_full_paths: list = []
    skipped = errors = 0

    for idx in tqdm(range(len(df)), desc="noise-precompute", ncols=100):
        row       = df.iloc[idx]
        face_path = str(row["face_crop_path"])
        stem      = _stable_stem(face_path)

        crop_pt = os.path.join(crops_dir, f"{stem}.pt")
        full_pt = os.path.join(fulls_dir, f"{stem}.pt")

        if not force and os.path.exists(crop_pt) and os.path.exists(full_pt):
            noise_crop_paths.append(crop_pt)
            noise_full_paths.append(full_pt)
            skipped += 1
            continue

        try:
            # ── STEP 1: run noise model on full frame (KNOWN_QUIRKS #3) ──
            src_path   = str(row["full_frame_path"]) if use_full_frame else face_path
            noise_full = _run_noiseprint(model, src_path, device)   # [1, H, W]

            # ── STEP 2: crop to face bbox ─────────────────────────────────
            H = noise_full.shape[1]
            W = noise_full.shape[2]

            if has_bbox and not pd.isna(row.get("x1", float("nan"))):
                bbox = (int(row["x1"]), int(row["y1"]),
                        int(row["x2"]), int(row["y2"]))
            else:
                bbox = (0, 0, W, H)

            noise_crop = crop_noise_to_face(noise_full, bbox, out_size=crop_size)

            torch.save(noise_crop, crop_pt)
            torch.save(noise_full, full_pt)

            noise_crop_paths.append(crop_pt)
            noise_full_paths.append(full_pt)

            del noise_full, noise_crop

        except Exception as exc:
            logger.warning("Noise error row %d (%s): %s", idx, face_path, exc)
            torch.save(torch.zeros(1, crop_size, crop_size), crop_pt)
            torch.save(torch.zeros(1, 299, 299), full_pt)
            noise_crop_paths.append(crop_pt)
            noise_full_paths.append(full_pt)
            errors += 1

        if idx % gc_interval == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    # ── Write updated manifest ─────────────────────────────────────────────
    out_df = add_noise_paths(df, noise_crop_paths, noise_full_paths)
    out_manifest = os.path.join(output_dir, "manifest.csv")
    out_df.to_csv(out_manifest, index=False)

    logger.info(
        "Noise precompute done → %s  (total=%d, skipped=%d, errors=%d)",
        out_manifest, len(df), skipped, errors,
    )
    return out_manifest
