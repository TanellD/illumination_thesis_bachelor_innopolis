"""
src/tsbi/shortcuts.py
=====================
T-SBI shortcut null variants N0–N6 (experiment C.5).

Each variant replaces one active step of the T-SBI pipeline with an identity
or trivial operation, then trains a one-epoch binary classifier and records
AUC. Separability is measured two-sided as |AUC − 0.5|.

  N0: JPEG roundtrip only (real crop → re-encode at T-SBI quality range)
  N1: src == tgt, full transfer (tests transfer MATH alone)
  N2: adjacent-frame pair (minimum temporal gap; tests boundary from two frames)
  N3: mask + feather only (tgt→tgt alpha-blend through mask; tests mask geometry)
  N4: align only (warp src to tgt box, alpha-paste, NO illumination transfer)
  N5_<mode>: single illumination mode on real temporal pairs (per-mode breakdown)
  N6: full T-SBI pipeline (headline number)

Ported from scripts/T_SBI/tsbi_shortcut.py.  The heavy training/eval loop is
the caller's responsibility; this module provides only the image-generation
kernels.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.tsbi.generator import (
    MODES,
    align_from_box,
    boxes_compatible,
    box_xyxy_to_xywh,
    crop_with_padding,
    detect_landmarks_68,
    ellipse_mask_on_box,
    hull_mask_from_landmarks,
    FACE_REGIONS,
    _HAS_MP,
)


# ── mask builder (shared) ─────────────────────────────────────────────────────

def _build_mask(
    t_frame: np.ndarray,
    t_box_xywh: Tuple[int, int, int, int],
    s_box_xywh: Tuple[int, int, int, int],
    rng: random.Random,
    no_landmarks: bool = False,
) -> Tuple[np.ndarray, str]:
    mask = None
    mask_kind = "ellipse"
    if _HAS_MP and not no_landmarks:
        lms = detect_landmarks_68(t_frame)
        if lms is not None:
            region = rng.choice(list(FACE_REGIONS.keys()))
            mask = hull_mask_from_landmarks(
                lms, t_frame.shape, region=region,
                feather_ratio=rng.uniform(0.05, 0.10),
            )
            mask_kind = f"hull:{region}"
    if mask is None:
        mask = ellipse_mask_on_box(
            t_box_xywh, t_frame.shape,
            feather_ratio=rng.uniform(0.18, 0.24),
            intersect=s_box_xywh,
        )
        mask_kind = "ellipse"
    return mask, mask_kind


def _read_frame(cap, idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def _crop_return(full: np.ndarray, box_xyxy: Tuple,
                 pad: float = 0.3) -> Optional[np.ndarray]:
    crop, _ = crop_with_padding(full, box_xyxy, pad=pad)
    return crop if (crop is not None and crop.size > 0) else None


# ── pair samplers ─────────────────────────────────────────────────────────────

def _pair_self(usable: list, rng: random.Random) -> Optional[Tuple]:
    if not usable:
        return None
    f = rng.choice(usable)
    return f, f


def _pair_adjacent(usable: list, rng: random.Random) -> Optional[Tuple]:
    if len(usable) < 2:
        return None
    s = sorted(usable, key=lambda u: u[0])
    i = rng.randrange(0, len(s) - 1)
    return s[i + 1], s[i]


def _pair_temporal(
    usable: list, rng: random.Random, fps: float,
    min_gap_sec: float = 0.8, max_gap_sec: float = 5.0,
) -> Optional[Tuple]:
    if len(usable) < 2:
        return None
    min_gap = max(1, int(min_gap_sec * fps))
    max_gap = max(min_gap + 1, int(max_gap_sec * fps))
    for _ in range(20):
        tgt = rng.choice(usable)
        cands = [
            u for u in usable
            if (min_gap <= abs(u[0] - tgt[0]) <= max_gap
                and boxes_compatible(box_xyxy_to_xywh(u[1]),
                                     box_xyxy_to_xywh(tgt[1])))
        ]
        if cands:
            return rng.choice(cands), tgt
    return None


# ── N0–N6 generators ──────────────────────────────────────────────────────────

def make_N0_roundtrip(
    cap, usable: list, rng: random.Random, pad: float = 0.3
) -> Optional[np.ndarray]:
    """Real target crop, no transfer — decode + crop + re-encode only."""
    if not usable:
        return None
    t_idx, t_box_xyxy = rng.choice(usable)[:2]
    t_frame = _read_frame(cap, t_idx)
    return _crop_return(t_frame, t_box_xyxy, pad) if t_frame is not None else None


def make_N1_self(
    cap, usable: list, rng: random.Random,
    modes: Optional[List[str]] = None, pad: float = 0.3,
    no_landmarks: bool = False,
) -> Optional[np.ndarray]:
    """src == tgt; full illumination transfer. Tests transfer math alone."""
    modes = modes or list(MODES.keys())
    pair = _pair_self(usable, rng)
    if pair is None:
        return None
    (s_idx, s_box_xyxy), (t_idx, t_box_xyxy) = pair
    t_frame = _read_frame(cap, t_idx)
    if t_frame is None:
        return None
    t_box = box_xyxy_to_xywh(t_box_xyxy)
    s_box = box_xyxy_to_xywh(s_box_xyxy)
    mask, _ = _build_mask(t_frame, t_box, s_box, rng, no_landmarks)
    src_ref = align_from_box(t_frame, s_box, t_box, t_frame.shape)
    try:
        fake = MODES[rng.choice(modes)](t_frame, src_ref, mask)
    except Exception:
        return None
    return _crop_return(fake, t_box_xyxy, pad)


def make_N2_adjacent(
    cap, usable: list, rng: random.Random, fps: float,
    modes: Optional[List[str]] = None, pad: float = 0.3,
    no_landmarks: bool = False,
) -> Optional[np.ndarray]:
    """src is the closest available frame to tgt. Minimum temporal gap."""
    modes = modes or list(MODES.keys())
    pair = _pair_adjacent(usable, rng)
    if pair is None:
        return None
    (s_idx, s_box_xyxy), (t_idx, t_box_xyxy) = pair
    s_frame = _read_frame(cap, s_idx)
    t_frame = _read_frame(cap, t_idx)
    if s_frame is None or t_frame is None:
        return None
    t_box = box_xyxy_to_xywh(t_box_xyxy)
    s_box = box_xyxy_to_xywh(s_box_xyxy)
    mask, _ = _build_mask(t_frame, t_box, s_box, rng, no_landmarks)
    src_ref = align_from_box(s_frame, s_box, t_box, t_frame.shape)
    try:
        fake = MODES[rng.choice(modes)](t_frame, src_ref, mask)
    except Exception:
        return None
    return _crop_return(fake, t_box_xyxy, pad)


def make_N3_mask_only(
    cap, usable: list, rng: random.Random, pad: float = 0.3,
    no_landmarks: bool = False,
) -> Optional[np.ndarray]:
    """Alpha-blend tgt with itself through the mask. No warp, no transfer."""
    if not usable:
        return None
    t_idx, t_box_xyxy = rng.choice(usable)[:2]
    t_frame = _read_frame(cap, t_idx)
    if t_frame is None:
        return None
    t_box = box_xyxy_to_xywh(t_box_xyxy)
    mask, _ = _build_mask(t_frame, t_box, t_box, rng, no_landmarks)
    a = (mask.astype(np.float32) / 255.0)[..., None]
    fake = np.clip(
        t_frame.astype(np.float32) * a + t_frame.astype(np.float32) * (1 - a),
        0, 255,
    ).astype(np.uint8)
    return _crop_return(fake, t_box_xyxy, pad)


def make_N4_align_only(
    cap, usable: list, rng: random.Random, fps: float,
    min_gap_sec: float = 0.8, max_gap_sec: float = 5.0,
    pad: float = 0.3, no_landmarks: bool = False,
) -> Optional[np.ndarray]:
    """Warp src to tgt's box, alpha-paste under mask. No illumination transfer."""
    pair = _pair_temporal(usable, rng, fps, min_gap_sec, max_gap_sec)
    if pair is None:
        return None
    (s_idx, s_box_xyxy), (t_idx, t_box_xyxy) = pair
    s_frame = _read_frame(cap, s_idx)
    t_frame = _read_frame(cap, t_idx)
    if s_frame is None or t_frame is None:
        return None
    t_box = box_xyxy_to_xywh(t_box_xyxy)
    s_box = box_xyxy_to_xywh(s_box_xyxy)
    mask, _ = _build_mask(t_frame, t_box, s_box, rng, no_landmarks)
    src_ref = align_from_box(s_frame, s_box, t_box, t_frame.shape)
    a = (mask.astype(np.float32) / 255.0)[..., None]
    fake = np.clip(
        src_ref.astype(np.float32) * a + t_frame.astype(np.float32) * (1 - a),
        0, 255,
    ).astype(np.uint8)
    return _crop_return(fake, t_box_xyxy, pad)


def make_N5_single_mode(
    cap, usable: list, rng: random.Random, fps: float,
    mode: str,
    min_gap_sec: float = 0.8, max_gap_sec: float = 5.0,
    pad: float = 0.3, no_landmarks: bool = False,
) -> Optional[np.ndarray]:
    """Single illumination mode on a real temporal pair."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}. Expected one of {list(MODES)}")
    pair = _pair_temporal(usable, rng, fps, min_gap_sec, max_gap_sec)
    if pair is None:
        return None
    (s_idx, s_box_xyxy), (t_idx, t_box_xyxy) = pair
    s_frame = _read_frame(cap, s_idx)
    t_frame = _read_frame(cap, t_idx)
    if s_frame is None or t_frame is None:
        return None
    t_box = box_xyxy_to_xywh(t_box_xyxy)
    s_box = box_xyxy_to_xywh(s_box_xyxy)
    mask, _ = _build_mask(t_frame, t_box, s_box, rng, no_landmarks)
    src_ref = align_from_box(s_frame, s_box, t_box, t_frame.shape)
    try:
        fake = MODES[mode](t_frame, src_ref, mask)
    except Exception:
        return None
    return _crop_return(fake, t_box_xyxy, pad)


def make_N6_full(
    cap, usable: list, rng: random.Random, fps: float,
    modes: Optional[List[str]] = None,
    min_gap_sec: float = 0.8, max_gap_sec: float = 5.0,
    pad: float = 0.3, no_landmarks: bool = False,
) -> Optional[np.ndarray]:
    """Full T-SBI pipeline — all modes, gap-constrained pair."""
    modes = modes or list(MODES.keys())
    pair = _pair_temporal(usable, rng, fps, min_gap_sec, max_gap_sec)
    if pair is None:
        return None
    (s_idx, s_box_xyxy), (t_idx, t_box_xyxy) = pair
    s_frame = _read_frame(cap, s_idx)
    t_frame = _read_frame(cap, t_idx)
    if s_frame is None or t_frame is None:
        return None
    t_box = box_xyxy_to_xywh(t_box_xyxy)
    s_box = box_xyxy_to_xywh(s_box_xyxy)
    mask, _ = _build_mask(t_frame, t_box, s_box, rng, no_landmarks)
    src_ref = align_from_box(s_frame, s_box, t_box, t_frame.shape)
    try:
        fake = MODES[rng.choice(modes)](t_frame, src_ref, mask)
    except Exception:
        return None
    return _crop_return(fake, t_box_xyxy, pad)


# Map variant name → generator function (for use in experiment entry points)
SHORTCUT_GENERATORS = {
    "N0": make_N0_roundtrip,
    "N1": make_N1_self,
    "N2": make_N2_adjacent,
    "N3": make_N3_mask_only,
    "N4": make_N4_align_only,
    **{f"N5_{m}": make_N5_single_mode for m in MODES},
    "N6": make_N6_full,
}
