"""
src/taxonomy/attributes.py
==========================
Per-frame attribute computation for the failure taxonomy (experiment F.1).

Computes nine image attributes per face crop:
  Tier 1 (OpenCV/NumPy, no heavy model):
    blur_var_lap, blur_bin
    L_mean, L_std, illum_bin, illum_harshness
    crop_padding_ratio, crop_tightness, touches_edge

  Tier 2 (MediaPipe FaceMesh — lazy-loaded per worker):
    face_detected
    yaw_deg, pitch_deg, roll_deg, pose_bin, pitch_bin
    ear_left, ear_right, ear_mean, eye_state
    iris_offset_norm, gaze_bin

Output: Parquet with one row per unique face_crop_path; stable schema.
Resumable: paths already in the Parquet are skipped.

Ported verbatim from scripts/FailureTaxonomy/taxonomy_analysis.py.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ── constants (must match eval preprocessing) ─────────────────────────────────

EVAL_INPUT_SIZE = 380

BLUR_BINS    = [(0, 50, "severe"), (50, 150, "moderate"),
                (150, 500, "mild"), (500, float("inf"), "sharp")]
JPEG_BINS    = [(0, 60, "low"), (60, 85, "mid"), (85, 101, "high")]
L_MEAN_BINS  = [(0, 80, "dark"), (80, 170, "normal"), (170, 256, "bright")]
L_STD_BINS   = [(0, 18, "flat"), (18, 35, "moderate"), (35, 256, "harsh")]
YAW_BINS     = [(0, 15, "frontal"), (15, 45, "three-quarter"), (45, 181, "profile")]

IRIS_OFFSET_AVERTED_THR = 0.10
EAR_CLOSED_THR          = 0.16
EAR_SQUINT_THR          = 0.22

# MediaPipe landmark indices
LM_NOSE_TIP        = 1
LM_CHIN            = 152
LM_LEFT_EYE_OUTER  = 33
LM_RIGHT_EYE_OUTER = 263
LM_LEFT_MOUTH      = 61
LM_RIGHT_MOUTH     = 291
LM_LEFT_EYE_EAR    = [33, 160, 158, 133, 153, 144]
LM_RIGHT_EYE_EAR   = [362, 385, 387, 263, 373, 380]
LM_LEFT_IRIS_CENTER  = 468
LM_RIGHT_IRIS_CENTER = 473
LM_LEFT_EYE_CORNERS  = (33, 133)
LM_RIGHT_EYE_CORNERS = (362, 263)

CANONICAL_3D = np.array([
    [0.0,    0.0,    0.0],
    [0.0,   -63.6,  -12.5],
    [-43.3,  32.7,  -26.0],
    [43.3,   32.7,  -26.0],
    [-28.9, -28.9,  -24.1],
    [28.9,  -28.9,  -24.1],
], dtype=np.float64)


# ── attribute schema ──────────────────────────────────────────────────────────

@dataclass
class CropAttrs:
    face_crop_path:   str   = ""
    decoded_ok:       bool  = False
    image_h:          int   = 0
    image_w:          int   = 0
    blur_var_lap:     float = float("nan")
    blur_bin:         str   = "unknown"
    jpeg_quality_est: float = float("nan")
    jpeg_bin:         str   = "unknown"
    crop_padding_ratio: float = float("nan")
    crop_tightness:   str   = "unknown"
    touches_edge:     bool  = False
    L_mean:           float = float("nan")
    L_std:            float = float("nan")
    illum_bin:        str   = "unknown"
    illum_harshness:  str   = "unknown"
    face_detected:    bool  = False
    yaw_deg:          float = float("nan")
    pitch_deg:        float = float("nan")
    roll_deg:         float = float("nan")
    pose_bin:         str   = "unknown"
    pitch_bin:        str   = "unknown"
    ear_left:         float = float("nan")
    ear_right:        float = float("nan")
    ear_mean:         float = float("nan")
    eye_state:        str   = "unknown"
    iris_offset_norm: float = float("nan")
    gaze_bin:         str   = "unknown"


# ── binning helper ────────────────────────────────────────────────────────────

def _bin(value: float, bins: list) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    for lo, hi, label in bins:
        if lo <= value < hi:
            return label
    return "unknown"


# ── Tier 1 extractors ─────────────────────────────────────────────────────────

def compute_blur(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_illumination(bgr: np.ndarray) -> Tuple[float, float]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    return float(L.mean()), float(L.std())


def compute_padding_and_edge(bgr: np.ndarray) -> Tuple[float, bool]:
    h, w = bgr.shape[:2]
    if h < 4 or w < 4:
        return float("nan"), False
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    borders = [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]
    pad_frame = 5
    if h >= 2 * pad_frame and w >= 2 * pad_frame:
        frame_mask = np.ones((h, w), dtype=bool)
        frame_mask[pad_frame:-pad_frame, pad_frame:-pad_frame] = False
        pad_ratio = float((gray[frame_mask] < 10).mean())
    else:
        pad_ratio = float((np.concatenate(borders) < 10).mean())
    edge_vars = [float(b.var()) for b in borders]
    touches_edge = any(v > 200 for v in edge_vars) and pad_ratio < 0.05
    return pad_ratio, touches_edge


def estimate_jpeg_quality(path: str) -> float:
    try:
        with Image.open(path) as im:
            if im.format != "JPEG":
                return float("nan")
            qtables = getattr(im, "quantization", None)
            if not qtables:
                return float("nan")
            std_lum = np.array([
                16, 11, 10, 16, 24, 40, 51, 61,
                12, 12, 14, 19, 26, 58, 60, 55,
                14, 13, 16, 24, 40, 57, 69, 56,
                14, 17, 22, 29, 51, 87, 80, 62,
                18, 22, 37, 56, 68, 109, 103, 77,
                24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101,
                72, 92, 95, 98, 112, 100, 103, 99,
            ], dtype=np.float32)
            file_lum = np.array(qtables[0], dtype=np.float32)
            if file_lum.size != std_lum.size:
                return float("nan")
            ratios  = file_lum / np.maximum(std_lum, 1e-6)
            scale   = float(np.median(ratios)) * 100.0
            if scale <= 0:
                return float("nan")
            q = (100 - scale / 2.0) if scale < 100 else (5000.0 / scale)
            return float(max(1.0, min(100.0, q)))
    except Exception:
        return float("nan")


def _crop_tightness(pad_ratio: float) -> str:
    if math.isnan(pad_ratio):
        return "unknown"
    if pad_ratio < 0.02:
        return "tight"
    if pad_ratio < 0.15:
        return "normal"
    if pad_ratio < 0.35:
        return "loose"
    return "extreme-pad"


# ── Tier 2 extractors — MediaPipe ─────────────────────────────────────────────

_face_mesh = None   # one per process


def _ensure_face_mesh():
    global _face_mesh
    if _face_mesh is not None:
        return _face_mesh
    import mediapipe as mp_
    _face_mesh = mp_.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
    )
    return _face_mesh


def _euler_from_rotmat(R: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy >= 1e-6:
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
        roll  = math.atan2(R[2, 1], R[2, 2])
    else:
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0
        roll  = math.atan2(-R[1, 2], R[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _solve_head_pose(lm_2d: np.ndarray, img_h: int,
                     img_w: int) -> Tuple[float, float, float]:
    focal = float(img_w)
    cam   = np.array([[focal, 0, img_w / 2.0],
                      [0, focal, img_h / 2.0],
                      [0,     0,         1.0]], dtype=np.float64)
    dist  = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(CANONICAL_3D, lm_2d, cam, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return float("nan"), float("nan"), float("nan")
    R, _ = cv2.Rodrigues(rvec)
    return _euler_from_rotmat(R)


def _ear(pts: np.ndarray) -> float:
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h  = np.linalg.norm(pts[0] - pts[3])
    return float((v1 + v2) / (2.0 * h)) if h >= 1e-6 else float("nan")


def _iris_offset(lm_xy: np.ndarray, iris_idx: int,
                 eye_corners: Tuple[int, int], ipd: float) -> float:
    if iris_idx >= lm_xy.shape[0] or ipd < 1e-6:
        return float("nan")
    iris = lm_xy[iris_idx]
    cL   = lm_xy[eye_corners[0]]
    cR   = lm_xy[eye_corners[1]]
    return float(np.linalg.norm(iris - (cL + cR) / 2.0) / ipd)


def compute_tier2(bgr: np.ndarray) -> dict:
    """Run MediaPipe FaceMesh and return Tier 2 attribute dict."""
    defaults = dict(
        face_detected=False,
        yaw_deg=float("nan"), pitch_deg=float("nan"), roll_deg=float("nan"),
        pose_bin="unknown", pitch_bin="unknown",
        ear_left=float("nan"), ear_right=float("nan"), ear_mean=float("nan"),
        eye_state="unknown",
        iris_offset_norm=float("nan"), gaze_bin="unknown",
    )
    try:
        fm  = _ensure_face_mesh()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return defaults
    except Exception:
        return defaults

    lm    = res.multi_face_landmarks[0].landmark
    h, w  = bgr.shape[:2]
    n_lm  = len(lm)
    lm_xy = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)

    # Head pose
    anchor_idxs = [LM_NOSE_TIP, LM_CHIN,
                   LM_LEFT_EYE_OUTER, LM_RIGHT_EYE_OUTER,
                   LM_LEFT_MOUTH, LM_RIGHT_MOUTH]
    if all(i < n_lm for i in anchor_idxs):
        lm_2d = np.array([lm_xy[i] for i in anchor_idxs], dtype=np.float64)
        yaw, pitch, roll = _solve_head_pose(lm_2d, h, w)
    else:
        yaw = pitch = roll = float("nan")

    pose_bin  = _bin(abs(yaw), YAW_BINS) if not math.isnan(yaw) else "unknown"
    pitch_bin = (
        "looking-down" if pitch < -10 else
        "looking-up"   if pitch >  10 else
        "level"
    ) if not math.isnan(pitch) else "unknown"

    # Eye aspect ratios
    ear_l = ear_r = float("nan")
    if all(i < n_lm for i in LM_LEFT_EYE_EAR):
        ear_l = _ear(np.array([lm_xy[i] for i in LM_LEFT_EYE_EAR]))
    if all(i < n_lm for i in LM_RIGHT_EYE_EAR):
        ear_r = _ear(np.array([lm_xy[i] for i in LM_RIGHT_EYE_EAR]))
    ear_mean = float((ear_l + ear_r) / 2.0) if not (
        math.isnan(ear_l) or math.isnan(ear_r)) else float("nan")
    eye_state = (
        "closed" if ear_mean < EAR_CLOSED_THR else
        "squint" if ear_mean < EAR_SQUINT_THR else
        "open"
    ) if not math.isnan(ear_mean) else "unknown"

    # Gaze (iris offset)
    ipd = float(np.linalg.norm(lm_xy[LM_LEFT_EYE_OUTER] - lm_xy[LM_RIGHT_EYE_OUTER])) \
          if n_lm > max(LM_LEFT_EYE_OUTER, LM_RIGHT_EYE_OUTER) else 0.0
    iris_off_l = _iris_offset(lm_xy, LM_LEFT_IRIS_CENTER, LM_LEFT_EYE_CORNERS, ipd)
    iris_off_r = _iris_offset(lm_xy, LM_RIGHT_IRIS_CENTER, LM_RIGHT_EYE_CORNERS, ipd)
    iris_off = (
        float((iris_off_l + iris_off_r) / 2.0)
        if not (math.isnan(iris_off_l) or math.isnan(iris_off_r))
        else float("nan")
    )
    gaze_bin = (
        "averted" if iris_off > IRIS_OFFSET_AVERTED_THR else "centered"
    ) if not math.isnan(iris_off) else "unknown"

    return dict(
        face_detected=True,
        yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
        pose_bin=pose_bin, pitch_bin=pitch_bin,
        ear_left=ear_l, ear_right=ear_r, ear_mean=ear_mean,
        eye_state=eye_state,
        iris_offset_norm=iris_off, gaze_bin=gaze_bin,
    )


# ── main extractor ────────────────────────────────────────────────────────────

def compute_attributes(path: str) -> CropAttrs:
    """Compute all attributes for a single face crop path."""
    attrs = CropAttrs(face_crop_path=str(path))
    bgr = cv2.imread(str(path))
    if bgr is None:
        return attrs
    attrs.decoded_ok = True
    attrs.image_h, attrs.image_w = bgr.shape[:2]

    # Blur
    attrs.blur_var_lap = compute_blur(bgr)
    attrs.blur_bin     = _bin(attrs.blur_var_lap, BLUR_BINS)

    # JPEG quality
    attrs.jpeg_quality_est = estimate_jpeg_quality(str(path))
    attrs.jpeg_bin         = _bin(attrs.jpeg_quality_est, JPEG_BINS)

    # Crop geometry
    pad, edge = compute_padding_and_edge(bgr)
    attrs.crop_padding_ratio = pad
    attrs.crop_tightness     = _crop_tightness(pad)
    attrs.touches_edge       = edge

    # Illumination
    attrs.L_mean, attrs.L_std = compute_illumination(bgr)
    attrs.illum_bin           = _bin(attrs.L_mean, L_MEAN_BINS)
    attrs.illum_harshness     = _bin(attrs.L_std,  L_STD_BINS)

    # Tier 2
    t2 = compute_tier2(bgr)
    for k, v in t2.items():
        setattr(attrs, k, v)

    return attrs


def compute_attributes_batch(
    paths: List[str],
    output_parquet: str,
    checkpoint_every: int = 5000,
    resume: bool = True,
) -> pd.DataFrame:
    """Compute attributes for a list of paths; write Parquet; resume-safe.

    Already-processed paths are skipped when resume=True and output_parquet
    already exists.
    """
    done: set = set()
    existing_rows: list = []
    if resume and os.path.exists(output_parquet):
        existing = pd.read_parquet(output_parquet)
        done = set(existing["face_crop_path"].tolist())
        existing_rows = existing.to_dict("records")

    todo = [p for p in paths if p not in done]
    new_rows: list = []

    for i, path in enumerate(todo):
        attrs = compute_attributes(path)
        new_rows.append(asdict(attrs))

        if checkpoint_every > 0 and (i + 1) % checkpoint_every == 0:
            df_checkpoint = pd.DataFrame(existing_rows + new_rows)
            df_checkpoint.to_parquet(output_parquet, index=False)

    all_rows = existing_rows + new_rows
    df = pd.DataFrame(all_rows)
    if len(df):
        df.to_parquet(output_parquet, index=False)
    return df
