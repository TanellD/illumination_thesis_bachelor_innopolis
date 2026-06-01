"""
Unit tests for src/data/manifest.py

Tests cover:
- validate_manifest rejects DataFrames with missing required columns
- validate_manifest rejects bad label values and bad split values
- validate_manifest rejects inverted bounding boxes
- validate_manifest catches directory-vs-file bug (KNOWN_QUIRKS #13)
- load_manifest round-trips through a temp CSV
- add_noise_paths inserts columns correctly
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from src.data.manifest import (
    MANIFEST_COLUMNS,
    ManifestValidationError,
    add_noise_paths,
    load_manifest,
    validate_manifest,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_valid_df(n: int = 4, tmp_dir: str = "") -> pd.DataFrame:
    """Return a minimal valid manifest DataFrame.

    If tmp_dir is given, face_crop_path entries are real empty files in that dir.
    """
    rows = []
    for i in range(n):
        path = ""
        if tmp_dir:
            path = os.path.join(tmp_dir, f"crop_{i}.jpg")
            open(path, "w").close()  # create empty file

        rows.append({
            "filename":       f"video_{i}_frame0000.jpg",
            "face_crop_path": path or f"/fake/path/crop_{i}.jpg",
            "label":          i % 2,
            "split":          ["train", "val", "test", "all"][i % 4],
            "method":         "original" if i % 2 == 0 else "Deepfakes",
            "video_id":       f"video_{i}",
            "frame_idx":      0,
            "x1": 10, "y1": 10, "x2": 90, "y2": 90,
            "confidence":     0.99,
            "dataset":        "ff++",
            "source":         "original",
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Column checks
# ──────────────────────────────────────────────────────────────────────────────

def test_valid_df_passes():
    df = _make_valid_df()
    validate_manifest(df, check_files_exist=False)  # no file check needed


def test_missing_column_raises():
    df = _make_valid_df()
    df = df.drop(columns=["label"])
    with pytest.raises(ManifestValidationError, match="Missing required columns"):
        validate_manifest(df, check_files_exist=False)


def test_multiple_missing_columns_raises():
    df = _make_valid_df()
    df = df.drop(columns=["label", "video_id", "dataset"])
    with pytest.raises(ManifestValidationError, match="Missing required columns"):
        validate_manifest(df, check_files_exist=False)


# ──────────────────────────────────────────────────────────────────────────────
# Label / split checks
# ──────────────────────────────────────────────────────────────────────────────

def test_bad_label_raises():
    df = _make_valid_df()
    df.loc[0, "label"] = 2
    with pytest.raises(ManifestValidationError, match="label values"):
        validate_manifest(df, check_files_exist=False)


def test_bad_split_raises():
    df = _make_valid_df()
    df.loc[0, "split"] = "holdout"
    with pytest.raises(ManifestValidationError, match="split values"):
        validate_manifest(df, check_files_exist=False)


# ──────────────────────────────────────────────────────────────────────────────
# Bounding box checks
# ──────────────────────────────────────────────────────────────────────────────

def test_inverted_bbox_x_raises():
    df = _make_valid_df()
    df.loc[0, "x1"] = 90
    df.loc[0, "x2"] = 10
    with pytest.raises(ManifestValidationError, match="bounding box"):
        validate_manifest(df, check_files_exist=False)


def test_equal_bbox_raises():
    df = _make_valid_df()
    df.loc[0, "y1"] = 50
    df.loc[0, "y2"] = 50
    with pytest.raises(ManifestValidationError, match="bounding box"):
        validate_manifest(df, check_files_exist=False)


# ──────────────────────────────────────────────────────────────────────────────
# File-existence check (KNOWN_QUIRKS #13)
# ──────────────────────────────────────────────────────────────────────────────

def test_file_existence_passes_with_real_files():
    with tempfile.TemporaryDirectory() as tmp:
        df = _make_valid_df(n=4, tmp_dir=tmp)
        validate_manifest(df, check_files_exist=True)


def test_missing_file_raises():
    df = _make_valid_df()
    df.loc[0, "face_crop_path"] = "/nonexistent/path/crop.jpg"
    with pytest.raises(ManifestValidationError, match="not files"):
        validate_manifest(df, check_files_exist=True)


def test_directory_path_raises():
    """A path pointing to a directory (not a file) must be caught — KNOWN_QUIRKS #13."""
    with tempfile.TemporaryDirectory() as tmp:
        df = _make_valid_df(n=1, tmp_dir=tmp)
        # Replace the file path with the directory itself
        df.loc[0, "face_crop_path"] = tmp
        with pytest.raises(ManifestValidationError, match="not files"):
            validate_manifest(df, check_files_exist=True)


# ──────────────────────────────────────────────────────────────────────────────
# Round-trip CSV
# ──────────────────────────────────────────────────────────────────────────────

def test_load_manifest_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        df_orig = _make_valid_df(n=4, tmp_dir=tmp)
        csv_path = os.path.join(tmp, "manifest.csv")
        df_orig.to_csv(csv_path, index=False)

        df_loaded = load_manifest(csv_path, check_files_exist=True)
        assert list(df_loaded.columns[:len(MANIFEST_COLUMNS)]) == MANIFEST_COLUMNS
        assert len(df_loaded) == 4


# ──────────────────────────────────────────────────────────────────────────────
# add_noise_paths
# ──────────────────────────────────────────────────────────────────────────────

def test_add_noise_paths():
    df = _make_valid_df(n=3)
    crops = [f"/noise/crops/{i}.pt" for i in range(3)]
    fulls = [f"/noise/fulls/{i}.pt" for i in range(3)]
    out   = add_noise_paths(df, crops, fulls)

    assert "noise_crop_path" in out.columns
    assert "noise_full_path"  in out.columns
    assert list(out["noise_crop_path"]) == crops
    assert list(out["noise_full_path"])  == fulls
    # original df not mutated
    assert "noise_crop_path" not in df.columns
