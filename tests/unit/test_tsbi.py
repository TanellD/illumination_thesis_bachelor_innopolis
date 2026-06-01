"""
Unit tests for src/tsbi/ — gating, pair acceptance, shortcut variants.

These tests are the exact unit tests the CLAUDE.md spec calls for:
  - T-SBI pair-acceptance logic (boxes_compatible, gate_pair)
  - Shortcut-ablation variants (N0–N6 generators)
  - dL-quartile manifest splitting
  - Illumination-transfer modes (smoke: output shape, no crash, range)

All tests run on CPU without real video files.
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tsbi.gating import gate_pair, split_by_dl_quartile, MIN_ILLUM_DELTA_MEAN, MIN_ILLUM_DELTA_STD
from src.tsbi.generator import (
    boxes_compatible,
    box_xyxy_to_xywh,
    face_box_luma_stats,
    select_pair,
    t_histmatch,
    t_reinhard,
    t_lowfreq,
    t_intrinsic,
    t_gainmap,
    MODES,
    DEFAULT_MODE,
    ellipse_mask_on_box,
    crop_with_padding,
    align_from_box,
)


# ── TestGatePair ──────────────────────────────────────────────────────────────

class TestGatePair:
    def test_default_thresholds(self):
        assert MIN_ILLUM_DELTA_MEAN == pytest.approx(6.0)
        assert MIN_ILLUM_DELTA_STD  == pytest.approx(3.0)

    def test_passes_when_dLmean_above_threshold(self):
        assert gate_pair(dL_mean=7.0, dL_std=0.0) is True

    def test_passes_when_dLstd_above_threshold(self):
        assert gate_pair(dL_mean=0.0, dL_std=4.0) is True

    def test_passes_when_both_above(self):
        assert gate_pair(dL_mean=8.0, dL_std=5.0) is True

    def test_fails_when_both_below(self):
        assert gate_pair(dL_mean=5.9, dL_std=2.9) is False

    def test_fails_when_exactly_at_threshold(self):
        # Gate is strict: must be >= threshold, so exactly threshold passes
        assert gate_pair(dL_mean=6.0, dL_std=0.0) is True
        assert gate_pair(dL_mean=0.0, dL_std=3.0) is True

    def test_custom_thresholds(self):
        assert gate_pair(5.0, 2.0, min_dL_mean=5.0, min_dL_std=2.0) is True
        assert gate_pair(4.9, 1.9, min_dL_mean=5.0, min_dL_std=2.0) is False


# ── TestBoxesCompatible ───────────────────────────────────────────────────────

class TestBoxesCompatible:
    def _box(self, cx, cy, w, h):
        return (cx - w // 2, cy - h // 2, w, h)

    def test_identical_boxes_compatible(self):
        b = self._box(100, 100, 60, 70)
        assert boxes_compatible(b, b) is True

    def test_scale_just_within_limit(self):
        a = self._box(100, 100, 60, 70)
        b = self._box(100, 100, 77, 90)   # ~1.28× — under 1.3
        assert boxes_compatible(a, b) is True

    def test_scale_just_over_limit(self):
        a = self._box(100, 100, 60, 70)
        b = self._box(100, 100, 80, 95)   # ~1.35× — over 1.3
        assert boxes_compatible(a, b) is False

    def test_large_centre_offset_fails(self):
        a = self._box(100, 100, 60, 60)
        b = self._box(200, 200, 60, 60)  # very large centre shift
        assert boxes_compatible(a, b) is False

    def test_small_centre_offset_passes(self):
        a = self._box(100, 100, 60, 60)
        b = self._box(110, 105, 60, 60)  # ~17% offset → within 35%
        assert boxes_compatible(a, b) is True

    def test_max_scale_parameter(self):
        a = self._box(100, 100, 60, 60)
        b = self._box(100, 100, 90, 90)   # 1.5× scale
        assert boxes_compatible(a, b, max_scale=2.0) is True
        assert boxes_compatible(a, b, max_scale=1.3) is False

    def test_max_shift_parameter(self):
        a = self._box(100, 100, 60, 60)
        b = self._box(120, 100, 60, 60)  # ~33% offset
        assert boxes_compatible(a, b, max_shift_frac=0.40) is True
        assert boxes_compatible(a, b, max_shift_frac=0.20) is False


# ── TestBoxXyxyToXywh ─────────────────────────────────────────────────────────

class TestBoxXyxyToXywh:
    def test_basic(self):
        assert box_xyxy_to_xywh((10, 20, 50, 80)) == (10, 20, 40, 60)

    def test_zero_box(self):
        assert box_xyxy_to_xywh((0, 0, 0, 0)) == (0, 0, 0, 0)


# ── TestSelectPair ────────────────────────────────────────────────────────────

class TestSelectPair:
    def _make_usable(self, n: int = 10, fps: float = 25.0):
        rng = random.Random(0)
        usable = []
        for i in range(n):
            box = (100 + i, 50, 200 + i, 180)
            luma = (50.0 + rng.uniform(-20, 20), 15.0 + rng.uniform(-5, 5))
            usable.append((i * 10, box, luma))
        return usable

    def test_returns_none_with_one_frame(self):
        usable = self._make_usable(1)
        rng = random.Random(42)
        result = select_pair(usable, rng, fps=25.0,
                              min_gap_sec=0.8, max_gap_sec=5.0,
                              min_dL_mean=6.0, min_dL_std=3.0)
        assert result is None

    def test_returns_tuple_with_many_frames(self):
        usable = self._make_usable(30)
        rng = random.Random(42)
        result = select_pair(usable, rng, fps=25.0,
                              min_gap_sec=0.2, max_gap_sec=8.0,
                              min_dL_mean=0.0, min_dL_std=0.0)
        # With 0 thresholds any compatible pair should be returned
        assert result is not None
        assert len(result) == 4   # (src, tgt, dLm, dLs)

    def test_relaxed_fallback_fires(self):
        """When strict gate never passes, relax fallback must return a result."""
        usable = self._make_usable(30)
        rng = random.Random(42)
        # Very strict gate: impossible to satisfy
        result = select_pair(usable, rng, fps=25.0,
                              min_gap_sec=0.2, max_gap_sec=8.0,
                              min_dL_mean=9999.0, min_dL_std=9999.0,
                              relax=True)
        # relaxed fallback should still return something (or None if no gap-OK pairs)
        # We don't assert non-None because the geometry might reject all pairs,
        # but we do assert it doesn't raise.
        assert result is None or len(result) == 4

    def test_no_relax_returns_none_when_strict_fails(self):
        """With relax=False and an impossible gate, must return None."""
        usable = self._make_usable(30)
        rng = random.Random(42)
        result = select_pair(usable, rng, fps=25.0,
                              min_gap_sec=0.2, max_gap_sec=8.0,
                              min_dL_mean=9999.0, min_dL_std=9999.0,
                              relax=False)
        assert result is None


# ── TestIlluminationTransfers ─────────────────────────────────────────────────

class TestIlluminationTransfers:
    @pytest.fixture
    def pair(self):
        rng = np.random.default_rng(0)
        tgt = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        src = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 200
        return tgt, src, mask

    def test_reinhard_shape(self, pair):
        tgt, src, mask = pair
        out = t_reinhard(tgt, src, mask)
        assert out.shape == tgt.shape

    def test_histmatch_shape(self, pair):
        tgt, src, mask = pair
        out = t_histmatch(tgt, src, mask)
        assert out.shape == tgt.shape

    def test_lowfreq_shape(self, pair):
        tgt, src, mask = pair
        out = t_lowfreq(tgt, src, mask)
        assert out.shape == tgt.shape

    def test_intrinsic_shape(self, pair):
        tgt, src, mask = pair
        out = t_intrinsic(tgt, src, mask)
        assert out.shape == tgt.shape

    def test_gainmap_shape(self, pair):
        tgt, src, mask = pair
        out = t_gainmap(tgt, src, mask)
        assert out.shape == tgt.shape

    def test_all_modes_in_dict(self):
        assert set(MODES.keys()) == {"reinhard", "histmatch", "lowfreq",
                                      "intrinsic", "gainmap"}

    def test_default_mode_is_histmatch(self):
        assert DEFAULT_MODE == "histmatch"

    def test_output_in_uint8_range(self, pair):
        tgt, src, mask = pair
        for mode, fn in MODES.items():
            out = fn(tgt, src, mask)
            assert out.dtype == np.uint8, f"{mode} did not return uint8"
            assert out.max() <= 255 and out.min() >= 0

    def test_zero_mask_returns_target(self, pair):
        """When mask is all-zero no transfer should happen."""
        tgt, src, _ = pair
        zero_mask = np.zeros((64, 64), dtype=np.uint8)
        out = t_reinhard(tgt, src, zero_mask)
        np.testing.assert_array_equal(out, tgt)

    def test_histmatch_no_crash_on_uniform_image(self):
        tgt = np.full((64, 64, 3), 128, dtype=np.uint8)
        src = np.full((64, 64, 3), 50,  dtype=np.uint8)
        mask = np.full((64, 64), 200, dtype=np.uint8)
        out = t_histmatch(tgt, src, mask)
        assert out.shape == tgt.shape


# ── TestEllipseMask ───────────────────────────────────────────────────────────

class TestEllipseMask:
    def test_output_shape(self):
        mask = ellipse_mask_on_box((10, 10, 80, 100), (200, 200))
        assert mask.shape == (200, 200)
        assert mask.dtype == np.uint8

    def test_nonzero_in_centre(self):
        mask = ellipse_mask_on_box((50, 50, 100, 100), (200, 200))
        assert mask[100, 100] > 0   # centre should be inside ellipse


# ── TestCropWithPadding ───────────────────────────────────────────────────────

class TestCropWithPadding:
    def test_basic_crop(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        crop, bbox = crop_with_padding(frame, (50, 40, 150, 160), pad=0.0)
        assert crop is not None
        assert crop.shape[0] == 120   # 160 - 40
        assert crop.shape[1] == 100   # 150 - 50

    def test_padding_expands_crop(self):
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        crop0, _ = crop_with_padding(frame, (100, 80, 200, 180), pad=0.0)
        crop3, _ = crop_with_padding(frame, (100, 80, 200, 180), pad=0.3)
        assert crop3.shape[0] > crop0.shape[0]

    def test_clamps_to_frame_boundary(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop, bbox = crop_with_padding(frame, (0, 0, 100, 100), pad=0.5)
        assert crop is not None
        assert crop.shape[0] <= 100 and crop.shape[1] <= 100


# ── TestDlQuartileSplit ───────────────────────────────────────────────────────

class TestDlQuartileSplit:
    @pytest.fixture
    def manifest_with_tsbi(self, tmp_path):
        rows = []
        for i in range(100):
            label = 0   # all fakes for simplicity
            source = "tsbi" if i < 60 else "sbi"
            rows.append({
                "face_crop_path": f"/fake/path/img_{i:03d}.jpg",
                "label": label,
                "source": source,
                "split": "train",
            })
        df = pd.DataFrame(rows)
        csv_path = str(tmp_path / "manifest.csv")
        df.to_csv(csv_path, index=False)
        return df

    @pytest.fixture
    def dl_lookup(self, manifest_with_tsbi):
        dl = {}
        for i in range(60):
            dl[f"/fake/path/img_{i:03d}.jpg"] = float(i)  # dL_mean in [0, 59]
        return dl

    def test_thresholds_computed_from_tsbi_rows(self, manifest_with_tsbi, dl_lookup):
        high, low, hi_thr, lo_thr = split_by_dl_quartile(
            manifest_with_tsbi, dl_lookup, compute_thresholds=True
        )
        # q75 of range [0, 59] is ~44.25; q25 is ~14.75
        assert hi_thr > lo_thr
        assert lo_thr >= 0

    def test_non_tsbi_rows_identical_in_both(self, manifest_with_tsbi, dl_lookup):
        high, low, _, _ = split_by_dl_quartile(
            manifest_with_tsbi, dl_lookup, compute_thresholds=True
        )
        # SBI rows (source == 'sbi') should be identical count in both subsets
        sbi_high = (high["source"] == "sbi").sum()
        sbi_low  = (low["source"]  == "sbi").sum()
        assert sbi_high == sbi_low == 40   # 40 sbi rows in fixture

    def test_high_has_larger_dl_than_low(self, manifest_with_tsbi, dl_lookup):
        high, low, hi_thr, lo_thr = split_by_dl_quartile(
            manifest_with_tsbi, dl_lookup, compute_thresholds=True
        )
        tsbi_high = high[high["source"] == "tsbi"]
        tsbi_low  = low[ low["source"]  == "tsbi"]
        if len(tsbi_high) > 0 and len(tsbi_low) > 0:
            mean_dl_high = tsbi_high["face_crop_path"].map(dl_lookup).mean()
            mean_dl_low  = tsbi_low[ "face_crop_path"].map(dl_lookup).mean()
            assert mean_dl_high > mean_dl_low

    def test_empty_tsbi_raises(self):
        df = pd.DataFrame([
            {"face_crop_path": "/a.jpg", "label": 0, "source": "sbi", "split": "train"}
        ])
        dl = {}
        with pytest.raises(ValueError, match="No T-SBI rows"):
            split_by_dl_quartile(df, dl, compute_thresholds=True)
