"""
Build the tiny-corpus fixture used by `make tiny` and integration tests.

Creates 8 real + 8 fake 64×64 synthetic face-crop JPEGs and a valid
manifest.csv.  The images are solid-colour patches — no real faces — purely
for exercising pipeline code paths without any real data dependency.

Usage:
  python tests/fixtures/build_tiny_corpus.py --out tests/fixtures/tiny_corpus/
"""

from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow required for build_tiny_corpus.py")


CROP_SIZE = 64   # tiny crops; real pipeline uses native bbox size or 299/380
SEED      = 42

MANIFEST_COLUMNS = [
    "filename", "face_crop_path", "label", "split", "method",
    "video_id", "frame_idx",
    "x1", "y1", "x2", "y2", "confidence",
    "dataset", "source",
]

# 8 real (label=0) + 8 fake (label=1)
_ENTRIES = [
    # (video_id, frame_idx, label, split, method, dataset, source)
    ("real_001", 0, 0, "train", "original",  "ff++", "original"),
    ("real_001", 5, 0, "train", "original",  "ff++", "original"),
    ("real_002", 0, 0, "val",   "original",  "ff++", "original"),
    ("real_002", 5, 0, "val",   "original",  "ff++", "original"),
    ("real_003", 0, 0, "test",  "original",  "ff++", "original"),
    ("real_003", 5, 0, "test",  "original",  "ff++", "original"),
    ("real_004", 0, 0, "train", "original",  "ff++", "original"),
    ("real_004", 5, 0, "train", "original",  "ff++", "original"),
    ("fake_001", 0, 1, "train", "Deepfakes", "ff++", "Deepfakes"),
    ("fake_001", 5, 1, "train", "Deepfakes", "ff++", "Deepfakes"),
    ("fake_002", 0, 1, "val",   "FaceSwap",  "ff++", "FaceSwap"),
    ("fake_002", 5, 1, "val",   "FaceSwap",  "ff++", "FaceSwap"),
    ("fake_003", 0, 1, "test",  "Deepfakes", "ff++", "Deepfakes"),
    ("fake_003", 5, 1, "test",  "Deepfakes", "ff++", "Deepfakes"),
    ("fake_004", 0, 1, "train", "FaceSwap",  "ff++", "FaceSwap"),
    ("fake_004", 5, 1, "train", "FaceSwap",  "ff++", "FaceSwap"),
]


def _synthetic_crop(label: int, rng: random.Random) -> np.ndarray:
    """Return an H×W×3 uint8 image: warm (real) or cool (fake) solid colour + noise."""
    if label == 0:
        base = (rng.randint(160, 220), rng.randint(120, 180), rng.randint(80, 140))
    else:
        base = (rng.randint(80, 140), rng.randint(120, 180), rng.randint(160, 220))

    img = np.full((CROP_SIZE, CROP_SIZE, 3), base, dtype=np.uint8)
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def build(out_dir: str) -> str:
    """Create the tiny corpus in out_dir.  Returns path to manifest.csv."""
    os.makedirs(out_dir, exist_ok=True)
    crops_dir = os.path.join(out_dir, "face_crops")
    os.makedirs(crops_dir, exist_ok=True)

    rng = random.Random(SEED)
    np.random.seed(SEED)

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()

        for video_id, frame_idx, label, split, method, dataset, source in _ENTRIES:
            img_name  = f"{video_id}_frame{frame_idx:04d}.jpg"
            crop_path = os.path.join(crops_dir, img_name)

            # Create synthetic JPEG
            crop_arr = _synthetic_crop(label, rng)
            Image.fromarray(crop_arr).save(crop_path, quality=95)

            writer.writerow({
                "filename":       img_name,
                "face_crop_path": crop_path,
                "label":          label,
                "split":          split,
                "method":         method,
                "video_id":       video_id,
                "frame_idx":      frame_idx,
                "x1": 5, "y1": 5,
                "x2": CROP_SIZE - 5, "y2": CROP_SIZE - 5,
                "confidence":     0.99,
                "dataset":        dataset,
                "source":         source,
            })

    print(f"Tiny corpus built: {len(_ENTRIES)} crops -> {manifest_path}")
    return manifest_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build tiny-corpus fixture")
    p.add_argument("--out", default="tests/fixtures/tiny_corpus/",
                   help="Output directory for the tiny corpus")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.out)
