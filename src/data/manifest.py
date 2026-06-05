"""
Manifest schema definition and validation for the thesis pipeline.

Every step that produces or consumes crops uses the same canonical column set.
Validation catches the directory-vs-file bug documented in KNOWN_QUIRKS #13.

Canonical columns
-----------------
filename        : basename of the crop JPEG (no directory path)
face_crop_path  : absolute path to the crop JPEG file
label           : 0 = real, 1 = fake
split           : 'train' | 'val' | 'test' | 'all'
method          : manipulation method or 'original' / detector name
video_id        : video stem (for frame-level grouping; see KNOWN_QUIRKS #12)
frame_idx       : integer frame index within the source video (0 for images)
x1, y1, x2, y2 : padded bounding box coordinates (integer pixels)
confidence      : MTCNN detection confidence [0, 1]
dataset         : dataset name ('ff++' | 'celebdf' | 'dfdc' | 'dff')
source          : grouping key for video-level aggregation (KNOWN_QUIRKS #12)

Optional columns (written only when noise precompute has run):
noise_crop_path : absolute path to the [1, 299, 299] .pt noise crop
noise_full_path : absolute path to the [1, H, W] .pt full-frame noise map
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────────────

MANIFEST_COLUMNS: List[str] = [
    "filename",
    "face_crop_path",
    "label",
    "split",
    "method",
    "video_id",
    "frame_idx",
    "x1", "y1", "x2", "y2",
    "confidence",
    "dataset",
    "source",
]

MANIFEST_OPTIONAL_COLUMNS: List[str] = [
    "noise_crop_path",
    "noise_full_path",
]

# Expected dtypes for numeric columns (used by validate_manifest)
_COLUMN_DTYPES = {
    "label":      "int",
    "frame_idx":  "int",
    "x1":         "int",
    "y1":         "int",
    "x2":         "int",
    "y2":         "int",
    "confidence": "float",
}

_VALID_SPLITS  = {"train", "val", "test", "all"}
_VALID_LABELS  = {0, 1}


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

class ManifestValidationError(ValueError):
    pass


def validate_manifest(
    df: pd.DataFrame,
    check_files_exist: bool = True,
    check_noise_paths: bool = False,
) -> None:
    """Validate a manifest DataFrame.  Raises ManifestValidationError on failure.

    Args:
        df:                The manifest DataFrame (already loaded from CSV).
        check_files_exist: If True, verify that every face_crop_path points to
                           an existing *file* (not a directory).  This catches
                           the KNOWN_QUIRKS #13 directory error.
        check_noise_paths: If True, also validate noise_crop_path / noise_full_path
                           columns if they are present.
    """
    errors: List[str] = []

    # ── Required columns ──────────────────────────────────────────
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    if errors:
        raise ManifestValidationError("\n".join(errors))

    # ── Label values ──────────────────────────────────────────────
    bad_labels = set(df["label"].unique()) - _VALID_LABELS
    if bad_labels:
        errors.append(f"Unexpected label values: {bad_labels}")

    # ── Split values ──────────────────────────────────────────────
    bad_splits = set(df["split"].unique()) - _VALID_SPLITS
    if bad_splits:
        errors.append(f"Unexpected split values: {bad_splits}")

    # ── Bounding box sanity ───────────────────────────────────────
    if not ((df["x2"] > df["x1"]).all() and (df["y2"] > df["y1"]).all()):
        errors.append("Some bounding boxes have x2 <= x1 or y2 <= y1")

    # ── File existence check (catches KNOWN_QUIRKS #13) ───────────
    if check_files_exist:
        bad_paths = []
        for path in df["face_crop_path"]:
            if not os.path.isfile(str(path)):
                bad_paths.append(path)
                if len(bad_paths) >= 5:  # report first 5 then stop
                    break
        if bad_paths:
            errors.append(
                f"{len(bad_paths)} face_crop_path entries are not files "
                f"(first few: {bad_paths[:3]}). "
                "Check KNOWN_QUIRKS #13 — paths must point to JPEG files, "
                "not to directories."
            )

    # ── Noise paths ───────────────────────────────────────────────
    if check_noise_paths:
        for col in ("noise_crop_path", "noise_full_path"):
            if col not in df.columns:
                errors.append(f"Noise column missing: {col}")
                continue
            bad = [p for p in df[col] if not os.path.isfile(str(p))]
            if bad:
                errors.append(
                    f"{len(bad)} entries in {col} are not files "
                    f"(first: {bad[:3]})"
                )

    if errors:
        raise ManifestValidationError("\n".join(errors))


def load_manifest(
    csv_path: str,
    check_files_exist: bool = False,
    check_noise_paths: bool = False,
) -> pd.DataFrame:
    """Load and optionally validate a manifest CSV.

    file-existence checks are off by default because the typical call
    happens before crops are available (e.g., to read labels for training).
    Pass check_files_exist=True before any evaluation run.
    """
    df = pd.read_csv(csv_path)
    validate_manifest(df, check_files_exist=check_files_exist,
                      check_noise_paths=check_noise_paths)
    return df


def add_noise_paths(df: pd.DataFrame, noise_crop_paths: list,
                    noise_full_paths: list) -> pd.DataFrame:
    """Return a copy of df with noise_crop_path and noise_full_path columns added."""
    out = df.copy()
    out["noise_crop_path"] = noise_crop_paths
    out["noise_full_path"] = noise_full_paths
    return out
