"""
src/sbi/generator.py
====================
Classic Self-Blended Images (SBI) generator (Shiohara & Yamasaki, CVPR 2022).

For each real face crop:
  1. Detect facial landmarks via MediaPipe FaceMesh (fallback: ellipse mask).
  2. Build a feathered convex-hull mask over a random face region.
  3. Optionally apply elastic deformation to the mask.
  4. Apply heavy photometric jitter (jitter_source_only) to ONE copy of the
     image (source), blend it back into the target through the mask.
  5. Save as JPEG (quality sampled from jpeg_quality_range=[75,98]).

Two jitter functions are intentionally separated:
  - jitter_mild:       safe to apply to BOTH classes during training
                       (no blur/resize fingerprint)
  - jitter_source_only: includes blur + downscale; ONLY on source copy

This split is load-bearing: if heavy jitter is applied to both classes at
training time (via albumentations), the model cannot learn
"was this image blurred?" as a class shortcut.  See KNOWN_QUIRKS.md.

Output CSV columns:
    filename, face_crop_path, label, split, method,
    x1, y1, x2, y2, confidence, source_real_path, jpeg_quality

Ported verbatim from scripts/T_SBI/generate_sbi.py.
"""
from __future__ import annotations

import csv
import gc
import os
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    _HAS_MP = True
except ImportError:
    _HAS_MP = False

# 468-point MediaPipe → 68-point dlib layout
_MP_TO_DLIB68 = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365, 397,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 6, 197, 195, 5, 4, 75, 1, 305,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146,
    78, 81, 13, 311, 308, 402, 14, 178,
]

FACE_REGIONS = {
    "full":  list(range(0, 27)),
    "eyes":  list(range(36, 48)) + list(range(17, 27)),
    "nose":  list(range(27, 36)),
    "mouth": list(range(48, 68)),
    "lower": list(range(4, 13)) + list(range(48, 68)),
}

SBI_CSV_COLS = [
    "filename", "face_crop_path", "label", "split", "method",
    "x1", "y1", "x2", "y2", "confidence", "source_real_path", "jpeg_quality",
]


# ── landmark detector ─────────────────────────────────────────────────────────

class LMDetector:
    """MediaPipe FaceMesh with proper per-batch cleanup to prevent leaks."""

    def __init__(self):
        self._mp_inst = None
        self._init()

    def _init(self):
        if not _HAS_MP:
            return
        if self._mp_inst is not None:
            self.cleanup()
        self._mp_inst = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )

    def detect(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        if self._mp_inst is None:
            return None
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = self._mp_inst.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0].landmark
            h, w = bgr.shape[:2]
            full = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
            return full[_MP_TO_DLIB68]
        except Exception:
            return None

    def cleanup(self) -> None:
        if self._mp_inst is not None:
            self._mp_inst.close()
            self._mp_inst = None
            gc.collect()

    def __del__(self):
        self.cleanup()


# ── mask helpers ──────────────────────────────────────────────────────────────

def hull_mask(landmarks: np.ndarray, shape, region: str,
              feather_ratio: float) -> np.ndarray:
    h, w = shape[:2]
    pts  = landmarks[FACE_REGIONS[region]].astype(np.int32)
    hull = cv2.convexHull(pts)
    m    = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    face_size = float(np.linalg.norm(landmarks[0] - landmarks[16]) + 1e-6)
    k = max(3, int(face_size * feather_ratio) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def ellipse_mask(shape, feather_ratio: float) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(m, (w // 2, int(h * 0.55)),
                (int(w * 0.35), int(h * 0.42)), 0, 0, 360, 255, -1)
    k = max(5, int(min(h, w) * feather_ratio) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def elastic_deform(mask: np.ndarray, strength: float) -> np.ndarray:
    h, w = mask.shape
    dx = cv2.GaussianBlur(
        (np.random.rand(h, w).astype(np.float32) - 0.5) * 2 * strength * w,
        (0, 0), sigmaX=w * 0.02)
    dy = cv2.GaussianBlur(
        (np.random.rand(h, w).astype(np.float32) - 0.5) * 2 * strength * h,
        (0, 0), sigmaX=h * 0.02)
    mx, my = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(mask, mx + dx, my + dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


# ── jitter ────────────────────────────────────────────────────────────────────

def jitter_mild(img: np.ndarray) -> np.ndarray:
    """Low-frequency colour jitter — safe on both classes (no resize trace)."""
    out = img.astype(np.float32)
    out = np.clip(out * random.uniform(0.9, 1.1) + random.uniform(-8, 8), 0, 255)
    hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + random.randint(-5, 5)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-15, 15), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def jitter_source_only(img: np.ndarray) -> np.ndarray:
    """Heavy jitter (includes blur + downscale).  Only call on the SOURCE copy."""
    out = jitter_mild(img).astype(np.float32)
    if random.random() < 0.5:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if random.random() < 0.5:
        h, w = out.shape[:2]
        s     = random.uniform(0.5, 0.9)
        small = cv2.resize(out, (int(w * s), int(h * s)),
                           interpolation=cv2.INTER_LINEAR)
        out   = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out.astype(np.uint8)


# ── blend ─────────────────────────────────────────────────────────────────────

def blend(target: np.ndarray, source: np.ndarray,
          mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "poisson":
        ys, xs = np.where(mask > 10)
        if len(xs):
            cx, cy = int(xs.mean()), int(ys.mean())
            binary = (mask > 10).astype(np.uint8) * 255
            try:
                return cv2.seamlessClone(source, target, binary, (cx, cy),
                                         cv2.NORMAL_CLONE)
            except cv2.error:
                pass
    m   = (mask.astype(np.float32) / 255.0)[..., None]
    out = source.astype(np.float32) * m + target.astype(np.float32) * (1 - m)
    return np.clip(out, 0, 255).astype(np.uint8)


# ── main generation kernel ────────────────────────────────────────────────────

def make_sbi(img: np.ndarray, detector: LMDetector) -> Optional[np.ndarray]:
    """Generate one SBI fake from a real face crop.  Returns None on failure."""
    lms = detector.detect(img)
    if lms is not None:
        region = random.choice(list(FACE_REGIONS.keys()))
        mask   = hull_mask(lms, img.shape, region,
                           feather_ratio=random.uniform(0.04, 0.09))
    else:
        mask = ellipse_mask(img.shape,
                             feather_ratio=random.uniform(0.10, 0.18))

    if random.random() < 0.5:
        mask = elastic_deform(mask, strength=random.uniform(0.01, 0.04))

    if random.random() < 0.5:
        source = jitter_source_only(img)
        target = img.copy()
    else:
        source = img.copy()
        target = jitter_source_only(img)

    mode = random.choice(["poisson", "alpha"])
    return blend(target, source, mask, mode)


# ── batch runner ──────────────────────────────────────────────────────────────

def generate_sbi_from_manifest(
    input_csv: str,
    out_dir: str,
    out_csv: str,
    splits: List[str] = ("train",),
    per_crop: int = 1,
    jpeg_quality_range: tuple = (75, 98),
    seed: int = 42,
    batch_size: int = 1000,
) -> int:
    """Generate SBI fakes from a real-crop manifest.

    Parameters
    ----------
    input_csv          : MTCNN manifest CSV (must have label, split,
                         face_crop_path columns)
    out_dir            : root directory for SBI JPEG crops
    out_csv            : output CSV path
    splits             : which splits to process (default: train only)
    per_crop           : SBI variants per real crop
    jpeg_quality_range : (min, max) JPEG quality to sample uniformly
    seed               : random seed
    batch_size         : detector re-initialised every this many crops

    Returns
    -------
    n_done : number of crops written
    """
    random.seed(seed)
    np.random.seed(seed)

    qmin, qmax = jpeg_quality_range

    rows = []
    with open(input_csv) as f:
        for row in csv.DictReader(f):
            if row["split"] in splits and int(row["label"]) == 0:
                rows.append(row)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fout = open(out_csv, "w", newline="")
    w    = csv.writer(fout)
    w.writerow(SBI_CSV_COLS)

    n_done = n_skip = 0

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start : batch_start + batch_size]
        detector = LMDetector()

        for row in batch:
            img = cv2.imread(row["face_crop_path"])
            if img is None:
                n_skip += 1
                continue

            for k in range(per_crop):
                out = None
                for _ in range(3):
                    try:
                        out = make_sbi(img, detector)
                        if out is not None:
                            break
                    except Exception:
                        out = None
                if out is None:
                    n_skip += 1
                    continue

                stem    = Path(row["filename"]).stem
                fname   = f"{stem}_sbi{k:02d}.jpg"
                out_sub = Path(out_dir) / row["split"]
                out_sub.mkdir(parents=True, exist_ok=True)
                out_path = out_sub / fname
                q_used   = random.randint(qmin, qmax)
                cv2.imwrite(str(out_path), out,
                            [cv2.IMWRITE_JPEG_QUALITY, q_used])

                w.writerow([
                    fname, str(out_path),
                    1,                         # label: fake
                    row["split"],
                    "sbi",
                    row["x1"], row["y1"], row["x2"], row["y2"],
                    row["confidence"],
                    row["face_crop_path"],
                    q_used,
                ])
                n_done += 1
                if n_done % 100 == 0:
                    gc.collect()

            del img
        detector.cleanup()

    fout.close()
    return n_done
