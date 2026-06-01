"""
src/tsbi/generator.py
=====================
Temporal Self-Blended Images (T-SBI) generator.

For each real training video:
  1. Scan frames at ~3 fps stride, detect faces with MTCNN, cache L-channel
     luminance stats over the face box.
  2. Select (src, tgt) pairs satisfying:
       - temporal gap in [min_gap_sec, max_gap_sec]
       - boxes_compatible(): scale ratio ≤ 1.3×, centre offset ≤ 35%
       - dL_mean ≥ min_illum_delta_mean  OR  dL_std ≥ min_illum_delta_std
       - If no pair passes the strict gate: pick the best-delta compatible
         pair anyway (relax fallback) and flag it in the CSV.
  3. Compute MediaPipe FaceMesh convex-hull mask on the target frame (6%
     feather ratio); fall back to an axis-aligned ellipse if landmarks fail.
  4. Apply one of the five illumination-transfer modes:
       histmatch (default), reinhard, lowfreq, intrinsic, gainmap.
  5. Crop the target face with padding_ratio=0.3 and save as JPEG (quality
     sampled uniformly from jpeg_quality_range=[75, 98]).
  6. Write one CSV row per saved crop with columns defined in TSBI_CSV_COLS.

Critical parameters (do NOT change defaults without updating KNOWN_QUIRKS.md):
  - max_scale_ratio: 1.3
  - max_centre_offset_frac: 0.35
  - min_illum_delta_mean: 6.0  (OR dL_std ≥ 3.0)
  - min_illum_delta_std: 3.0
  - feather_ratio: 0.06
  - padding_ratio: 0.3

Ported verbatim from scripts/T_SBI/generate_tsbi.py.
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    _HAS_MP = True
except ImportError:
    _HAS_MP = False

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Output CSV column contract — must match MANIFEST_COLUMNS in manifest.py
# plus T-SBI-specific extras.
TSBI_CSV_COLS = [
    "filename", "face_crop_path", "label", "split", "method",
    "x1", "y1", "x2", "y2", "confidence",
    "src_frame", "tgt_frame", "tsbi_mode", "source_video",
    "jpeg_quality", "mask_kind",
    "illum_delta_L", "illum_delta_Lstd", "illum_relaxed",
]

# Landmark index map: MediaPipe 468-point → dlib-68-point layout
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
    "full":     list(range(0, 27)),
    "cheek_fh": list(range(0, 27)),
    "lower":    list(range(4, 13)) + list(range(48, 68)),
    "upper":    list(range(17, 27)) + list(range(36, 48)),
}


# ── MediaPipe singleton ───────────────────────────────────────────────────────

class _FaceMeshSingleton:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None and _HAS_MP:
            cls._inst = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
            )
        return cls._inst

    @classmethod
    def cleanup(cls) -> None:
        if cls._inst is not None:
            cls._inst.close()
            cls._inst = None


# ── landmark and mask helpers ─────────────────────────────────────────────────

def detect_landmarks_68(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Return (68, 2) landmark coords in pixel space, or None."""
    fm = _FaceMeshSingleton.get()
    if fm is None:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    h, w = img_bgr.shape[:2]
    full = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
    if full.shape[0] < max(_MP_TO_DLIB68) + 1:
        return None
    return full[_MP_TO_DLIB68]


def hull_mask_from_landmarks(
    landmarks: np.ndarray,
    image_shape: Tuple[int, int],
    region: str = "full",
    feather_ratio: float = 0.06,
) -> np.ndarray:
    """Feathered convex-hull mask over the requested landmark region."""
    h, w = image_shape[:2]
    pts = landmarks[FACE_REGIONS[region]].astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    face_size = float(np.linalg.norm(landmarks[0] - landmarks[16]) + 1e-6)
    k = max(3, int(face_size * feather_ratio) | 1)
    return cv2.GaussianBlur(mask, (k, k), 0)


def ellipse_mask_on_box(
    box: Tuple[int, int, int, int],
    shape: Tuple[int, int],
    shrink: float = 0.75,
    feather_ratio: float = 0.20,
    y_off: float = 0.08,
    intersect: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Fallback elliptical mask when landmark detection fails."""
    x, y, w, h = box
    H, W = shape[:2]
    if intersect is not None:
        ix, iy, iw, ih = intersect
        x0 = max(x, ix); y0 = max(y, iy)
        x1 = min(x + w, ix + iw); y1 = min(y + h, iy + ih)
        if x1 > x0 and y1 > y0:
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
    cx = x + w // 2
    cy = int(y + h * (0.5 + y_off))
    ax = int(w * 0.5 * shrink)
    ay = int(h * 0.55 * shrink)
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    k = max(5, int(max(w, h) * feather_ratio) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


# ── pair geometry ─────────────────────────────────────────────────────────────

def boxes_compatible(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
    max_scale: float = 1.3,
    max_shift_frac: float = 0.35,
) -> bool:
    """True when two face boxes are geometrically compatible for T-SBI pairing.

    max_scale and max_shift_frac must NOT be changed — they are the pair-
    acceptance parameters that produced the reported results.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    s = max(aw / max(1, bw), bw / max(1, aw),
            ah / max(1, bh), bh / max(1, ah))
    if s > max_scale:
        return False
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    ref = (aw + bw + ah + bh) * 0.25
    return (abs(acx - bcx) <= max_shift_frac * ref
            and abs(acy - bcy) <= max_shift_frac * ref)


def box_xyxy_to_xywh(b: Tuple) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


# ── luminance stats ───────────────────────────────────────────────────────────

def face_box_luma_stats(
    frame_bgr: np.ndarray,
    box_xyxy: Tuple,
) -> Optional[Tuple[float, float]]:
    """Return (L_mean, L_std) inside the face box, or None if box is degenerate."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    H, W = frame_bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    return float(L.mean()), float(L.std())


# ── illumination transfer modes ───────────────────────────────────────────────

def _masked_mean_std(
    img_lab: np.ndarray, mask_bin: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    m = mask_bin.astype(bool)
    if m.sum() < 10:
        return None, None
    px = img_lab[m]
    return px.mean(axis=0), px.std(axis=0) + 1e-6


def _cdf_lut(
    src_vals: np.ndarray, tgt_vals: np.ndarray,
    trim: float = 2.0, max_shift: int = 40,
) -> np.ndarray:
    if trim > 0 and len(src_vals) > 50 and len(tgt_vals) > 50:
        lo_s, hi_s = np.percentile(src_vals, [trim, 100 - trim])
        lo_t, hi_t = np.percentile(tgt_vals, [trim, 100 - trim])
        src_vals = src_vals[(src_vals >= lo_s) & (src_vals <= hi_s)]
        tgt_vals = tgt_vals[(tgt_vals >= lo_t) & (tgt_vals <= hi_t)]
        if len(src_vals) < 10 or len(tgt_vals) < 10:
            return np.arange(256, dtype=np.uint8)
    sh, _ = np.histogram(src_vals, bins=256, range=(0, 256))
    th, _ = np.histogram(tgt_vals, bins=256, range=(0, 256))
    sc = np.cumsum(sh).astype(np.float64)
    tc = np.cumsum(th).astype(np.float64)
    if sc[-1] < 1 or tc[-1] < 1:
        return np.arange(256, dtype=np.uint8)
    sc /= sc[-1]; tc /= tc[-1]
    lut = np.interp(tc, sc, np.arange(256))
    ident = np.arange(256, dtype=np.float64)
    lut = np.clip(lut, ident - max_shift, ident + max_shift)
    return np.clip(lut, 0, 255).astype(np.uint8)


def t_reinhard(
    tgt: np.ndarray, src: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    mb = (mask > 10).astype(np.uint8)
    if mb.sum() < 10:
        return tgt.copy()
    tL = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB).astype(np.float32)
    sL = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    tm, ts = _masked_mean_std(tL, mb)
    sm, ss = _masked_mean_std(sL, mb)
    if tm is None or sm is None:
        return tgt.copy()
    sh = (tL - tm) * (ss / ts) + sm
    a = (mask.astype(np.float32) / 255.0)[..., None]
    out = sh * a + tL * (1 - a)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def t_histmatch(
    tgt: np.ndarray, src: np.ndarray, mask: np.ndarray,
    strength: float = 0.55, env_sigma_ratio: float = 0.04,
) -> np.ndarray:
    mb = (mask > 10).astype(np.uint8)
    if mb.sum() < 10:
        return tgt.copy()
    tL = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB)
    sL = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    h, w = tgt.shape[:2]
    sigma = max(2.0, min(h, w) * env_sigma_ratio)
    k = int(sigma * 6) | 1
    tf_ = cv2.GaussianBlur(tL, (k, k), sigma)
    sf_ = cv2.GaussianBlur(sL, (k, k), sigma)
    matched = tL.copy()
    idx = np.where(mb > 0)
    tvals = tf_[..., 0][idx]
    svals = sf_[..., 0][mb > 0]
    if len(tvals) and len(svals):
        lut = _cdf_lut(svals, tvals)
        matched[..., 0][idx] = lut[tL[..., 0][idx]]
    m = matched.astype(np.float32)
    t = tL.astype(np.float32)
    soft = (mask.astype(np.float32) / 255.0)[..., None] * strength
    out = m * soft + t * (1 - soft)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def t_lowfreq(
    tgt: np.ndarray, src_w: np.ndarray, mask: np.ndarray,
    sigma_ratio: float = 0.05,
) -> np.ndarray:
    h, w = tgt.shape[:2]
    sigma = max(3.0, min(h, w) * sigma_ratio)
    k = int(sigma * 6) | 1
    tL = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB).astype(np.float32)
    sL = cv2.cvtColor(src_w, cv2.COLOR_BGR2LAB).astype(np.float32)
    tl = cv2.GaussianBlur(tL, (k, k), sigma)
    sl = cv2.GaussianBlur(sL, (k, k), sigma)
    th_ = tL - tl
    cw = 0.5
    c = np.empty_like(tL)
    c[..., 0] = sl[..., 0] + th_[..., 0]
    c[..., 1] = cw * sl[..., 1] + (1 - cw) * tl[..., 1] + th_[..., 1]
    c[..., 2] = cw * sl[..., 2] + (1 - cw) * tl[..., 2] + th_[..., 2]
    a = (mask.astype(np.float32) / 255.0)[..., None]
    out = c * a + tL * (1 - a)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def t_intrinsic(
    tgt: np.ndarray, src_w: np.ndarray, mask: np.ndarray,
    sigma_ratio: float = 0.04,
) -> np.ndarray:
    h, w = tgt.shape[:2]
    sigma = max(3.0, min(h, w) * sigma_ratio)
    k = int(sigma * 6) | 1
    tf_ = tgt.astype(np.float32) + 1.0
    tl = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
    sl = cv2.cvtColor(src_w, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
    St = cv2.GaussianBlur(tl, (k, k), sigma)
    Ss = cv2.GaussianBlur(sl, (k, k), sigma)
    Rt = tf_ / St[..., None]
    relit = Rt * Ss[..., None]
    a = (mask.astype(np.float32) / 255.0)[..., None]
    out = relit * a + tf_ * (1 - a)
    return np.clip(out - 1.0, 0, 255).astype(np.uint8)


def t_gainmap(
    tgt: np.ndarray, src_w: np.ndarray, mask: np.ndarray,
    sigma_ratio: float = 0.06,
) -> np.ndarray:
    h, w = tgt.shape[:2]
    sigma = max(3.0, min(h, w) * sigma_ratio)
    k = int(sigma * 6) | 1
    tf_ = tgt.astype(np.float32) + 1.0
    sf = src_w.astype(np.float32) + 1.0
    tb = cv2.GaussianBlur(tf_, (k, k), sigma)
    sb = cv2.GaussianBlur(sf, (k, k), sigma)
    gain = np.clip(sb / tb, 0.6, 1.6)
    relit = tf_ * gain
    a = (mask.astype(np.float32) / 255.0)[..., None]
    out = relit * a + tf_ * (1 - a)
    return np.clip(out - 1.0, 0, 255).astype(np.uint8)


MODES = {
    "reinhard":  t_reinhard,
    "histmatch": t_histmatch,
    "lowfreq":   t_lowfreq,
    "intrinsic": t_intrinsic,
    "gainmap":   t_gainmap,
}

DEFAULT_MODE = "histmatch"


# ── affine alignment ──────────────────────────────────────────────────────────

def align_from_box(
    src_img: np.ndarray,
    src_box: Tuple[int, int, int, int],
    tgt_box: Tuple[int, int, int, int],
    tgt_shape: Tuple[int, int],
) -> np.ndarray:
    sx, sy, sw, sh = src_box
    tx, ty, tw, th = tgt_box
    scx, scy = sx + sw / 2.0, sy + sh / 2.0
    tcx, tcy = tx + tw / 2.0, ty + th / 2.0
    scale = (tw / max(1, sw) + th / max(1, sh)) * 0.5
    M = np.array(
        [[scale, 0, tcx - scale * scx],
         [0, scale, tcy - scale * scy]],
        dtype=np.float32,
    )
    H, W = tgt_shape[:2]
    return cv2.warpAffine(src_img, M, (W, H),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


# ── crop helper ───────────────────────────────────────────────────────────────

def crop_with_padding(
    frame_bgr: np.ndarray,
    box: Tuple,
    pad: float = 0.3,
) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    x1p = max(0, int(x1) - px)
    y1p = max(0, int(y1) - py)
    x2p = min(w, int(x2) + px)
    y2p = min(h, int(y2) + py)
    crop = frame_bgr[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return None, (x1p, y1p, x2p, y2p)
    return crop, (x1p, y1p, x2p, y2p)


# ── pair selection ────────────────────────────────────────────────────────────

def select_pair(
    usable_with_luma: list,
    rng: random.Random,
    fps: float,
    min_gap_sec: float,
    max_gap_sec: float,
    min_dL_mean: float,
    min_dL_std: float,
    relax: bool = True,
) -> Optional[Tuple]:
    """Select a (src, tgt, dL_mean, dL_std) pair from frame list.

    usable_with_luma: list of (frame_idx, box_xyxy, (Lmean, Lstd))
    Returns None if no compatible pair found after 20 attempts.
    """
    if len(usable_with_luma) < 2:
        return None
    min_gap = max(1, int(min_gap_sec * fps))
    max_gap = max(min_gap + 1, int(max_gap_sec * fps))

    for _ in range(20):
        tgt = rng.choice(usable_with_luma)
        t_idx, t_box, (tLm, tLs) = tgt
        gap_ok = []
        for u in usable_with_luma:
            u_idx, u_box, _ = u
            if not (min_gap <= abs(u_idx - t_idx) <= max_gap):
                continue
            if not boxes_compatible(box_xyxy_to_xywh(u_box),
                                    box_xyxy_to_xywh(t_box)):
                continue
            gap_ok.append(u)
        if not gap_ok:
            continue

        strict = []
        for u in gap_ok:
            _, _, (uLm, uLs) = u
            dLm = abs(uLm - tLm)
            dLs = abs(uLs - tLs)
            if dLm >= min_dL_mean or dLs >= min_dL_std:
                strict.append((u, dLm, dLs))
        if strict:
            u, dLm, dLs = rng.choice(strict)
            return u, tgt, dLm, dLs
        if relax and gap_ok:
            scored = []
            for u in gap_ok:
                _, _, (uLm, uLs) = u
                dLm = abs(uLm - tLm)
                dLs = abs(uLs - tLs)
                scored.append((max(dLm, dLs), u, dLm, dLs))
            scored.sort(reverse=True, key=lambda x: x[0])
            _, u, dLm, dLs = scored[0]
            return u, tgt, dLm, dLs
    return None
