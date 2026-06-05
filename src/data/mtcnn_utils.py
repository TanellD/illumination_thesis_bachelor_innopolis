"""
Shared MTCNN initialisation and face-crop helpers.

All four dataset extraction scripts used the same MTCNN parameters:
  keep_all=True, thresholds=[0.6, 0.7, 0.7], min_face_size=20
and the same padding logic (symmetric per-axis padding of the bbox).
This module centralises those so dataset-specific scripts stay thin.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# MTCNN parameters that were constant across all original scripts.
_MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]
_MTCNN_MIN_FACE   = 20


def build_mtcnn(device: Optional[torch.device] = None):
    """Return a configured MTCNN instance.

    Lazy-imports facenet_pytorch so the module can be imported on machines
    without it installed (e.g., during analysis-only runs).
    """
    from facenet_pytorch import MTCNN  # type: ignore

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return MTCNN(
        keep_all=True,
        thresholds=_MTCNN_THRESHOLDS,
        post_process=True,
        device=device,
        select_largest=False,
        min_face_size=_MTCNN_MIN_FACE,
    )


def pick_best_face(
    boxes: Optional[np.ndarray],
    probs: Optional[np.ndarray],
    min_conf: float = 0.85,
) -> Optional[Tuple[np.ndarray, float]]:
    """Return (box, confidence) for the largest face above min_conf, or None.

    "Largest" is by bbox area, matching the original FF++ script behaviour.
    CelebDF used argmax(probs) instead; that is preserved in its extractor.
    """
    if boxes is None or len(boxes) == 0 or probs is None:
        return None

    best = None
    best_area = 0.0
    for box, p in zip(boxes, probs):
        if float(p) < min_conf:
            continue
        x1, y1, x2, y2 = box
        area = float((x2 - x1) * (y2 - y1))
        if area > best_area:
            best_area = area
            best = (box, float(p))
    return best


def crop_with_padding(
    frame_bgr: np.ndarray,
    box: np.ndarray,
    padding_ratio: float = 0.3,
    target_size: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], int, int, int, int]:
    """Crop the face region from a BGR frame with symmetric padding.

    Args:
        frame_bgr:     Full-frame BGR image from OpenCV.
        box:           MTCNN bounding box [x1, y1, x2, y2].
        padding_ratio: Fractional padding applied symmetrically to each axis.
                       FF++ used 0.3; CelebDF/DFDC/DFF used 0.25.
        target_size:   If given, resize crop to (target_size × target_size).
                       Pass None to keep native bbox size (FF++ behaviour).

    Returns:
        (crop_bgr or None, x1_padded, y1_padded, x2_padded, y2_padded)
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    x1p = max(0, int(x1) - pad_x)
    y1p = max(0, int(y1) - pad_y)
    x2p = min(w, int(x2) + pad_x)
    y2p = min(h, int(y2) + pad_y)

    crop = frame_bgr[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return None, x1p, y1p, x2p, y2p

    if target_size is not None:
        crop = cv2.resize(crop, (target_size, target_size))

    return crop, x1p, y1p, x2p, y2p


def detect_faces_batch(
    mtcnn,
    frames_rgb: list,
    batch_size: int = 8,
) -> list:
    """Run MTCNN on a list of RGB numpy arrays, returning one result per frame.

    Each result is either (boxes, probs) arrays or (None, None) on failure.
    Uses fallback one-by-one detection if the batch call raises.
    """
    results = [(None, None)] * len(frames_rgb)

    for start in range(0, len(frames_rgb), batch_size):
        batch     = frames_rgb[start : start + batch_size]
        pil_batch = [Image.fromarray(f) for f in batch]

        try:
            batch_boxes, batch_probs = mtcnn.detect(pil_batch)
            for i, (boxes, probs) in enumerate(zip(batch_boxes, batch_probs)):
                results[start + i] = (boxes, probs)
        except Exception:
            for i, pil_img in enumerate(pil_batch):
                try:
                    boxes, probs = mtcnn.detect([pil_img])
                    results[start + i] = (
                        boxes[0] if boxes is not None else None,
                        probs[0] if probs is not None else None,
                    )
                except Exception:
                    pass

    return results


def find_first_face_frame(
    cap,
    mtcnn,
    max_check: int = 60,
    batch_size: int = 8,
    min_conf: float = 0.85,
) -> Optional[int]:
    """Return the index of the first frame that contains a detectable face.

    Reads up to max_check frames from cap (seeking explicitly), batches
    them for MTCNN, and returns the earliest index where confidence ≥ min_conf.
    """
    frames:  list = []
    indices: list = []

    for idx in range(max_check):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        indices.append(idx)

        if len(frames) >= batch_size or idx == max_check - 1:
            detections = detect_faces_batch(mtcnn, frames, batch_size=batch_size)
            for i, (boxes, probs) in enumerate(detections):
                result = pick_best_face(boxes, probs, min_conf=min_conf)
                if result is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return indices[i]
            frames, indices = [], []

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return None
