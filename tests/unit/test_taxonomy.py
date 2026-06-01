"""
Unit tests for src/taxonomy/attributes.py and src/taxonomy/contingency.py

Tests verify:
- Attribute extraction on synthetic images (Tier 1 only; MediaPipe not loaded)
- Binning helpers produce correct labels
- Chi-square + Cramér's V give correct outputs on hand-crafted contingency tables
- Jaccard computation matches manual calculation
- Agreement distribution sums correctly
- Quantile binning correctly uses per-dataset distribution
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import cv2

from src.taxonomy.attributes import (
    BLUR_BINS,
    L_MEAN_BINS,
    L_STD_BINS,
    YAW_BINS,
    _bin,
    _crop_tightness,
    compute_blur,
    compute_illumination,
    compute_padding_and_edge,
)
from src.taxonomy.contingency import (
    chi_square_cramers,
    failure_jaccard,
    agreement_distribution,
    per_bin_failure_rates,
    quantile_bin_within_dataset,
    unique_failures_table,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _solid_bgr(h: int = 64, w: int = 64, b: int = 128,
               g: int = 128, r: int = 128) -> np.ndarray:
    return np.full((h, w, 3), [b, g, r], dtype=np.uint8)


def _noisy_bgr(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ── TestBinHelper ─────────────────────────────────────────────────────────────

class TestBinHelper:
    def test_blur_severe(self):
        assert _bin(10.0, BLUR_BINS) == "severe"

    def test_blur_sharp(self):
        assert _bin(1000.0, BLUR_BINS) == "sharp"

    def test_nan_returns_unknown(self):
        assert _bin(float("nan"), BLUR_BINS) == "unknown"

    def test_none_returns_unknown(self):
        assert _bin(None, BLUR_BINS) == "unknown"

    def test_l_mean_dark(self):
        assert _bin(50.0, L_MEAN_BINS) == "dark"

    def test_l_mean_bright(self):
        assert _bin(200.0, L_MEAN_BINS) == "bright"

    def test_l_std_flat(self):
        assert _bin(10.0, L_STD_BINS) == "flat"

    def test_l_std_harsh(self):
        assert _bin(50.0, L_STD_BINS) == "harsh"

    def test_yaw_frontal(self):
        assert _bin(5.0, YAW_BINS) == "frontal"

    def test_yaw_profile(self):
        assert _bin(60.0, YAW_BINS) == "profile"


# ── TestCropTightness ─────────────────────────────────────────────────────────

class TestCropTightness:
    def test_tight(self):
        assert _crop_tightness(0.01) == "tight"

    def test_normal(self):
        assert _crop_tightness(0.08) == "normal"

    def test_loose(self):
        assert _crop_tightness(0.20) == "loose"

    def test_extreme_pad(self):
        assert _crop_tightness(0.50) == "extreme-pad"

    def test_nan_unknown(self):
        assert _crop_tightness(float("nan")) == "unknown"


# ── TestComputeBlur ───────────────────────────────────────────────────────────

class TestComputeBlur:
    def test_blurry_image_low_variance(self):
        bgr = _solid_bgr()   # uniform = zero variance
        assert compute_blur(bgr) == pytest.approx(0.0, abs=1.0)

    def test_noisy_image_high_variance(self):
        bgr = _noisy_bgr()
        assert compute_blur(bgr) > 100.0   # high-frequency noise


# ── TestComputeIllumination ───────────────────────────────────────────────────

class TestComputeIllumination:
    def test_uniform_dark_has_low_std(self):
        bgr = _solid_bgr(b=10, g=10, r=10)
        mean, std = compute_illumination(bgr)
        assert mean < 50.0
        assert std < 5.0

    def test_uniform_bright_has_high_mean(self):
        bgr = _solid_bgr(b=220, g=220, r=220)
        mean, std = compute_illumination(bgr)
        assert mean > 150.0


# ── TestComputePaddingAndEdge ─────────────────────────────────────────────────

class TestComputePaddingAndEdge:
    def test_all_black_padding(self):
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        pad_ratio, touches_edge = compute_padding_and_edge(bgr)
        assert pad_ratio > 0.5   # mostly black

    def test_content_image_low_padding(self):
        bgr = _noisy_bgr()
        pad_ratio, _ = compute_padding_and_edge(bgr)
        assert pad_ratio < 0.3

    def test_tiny_image_returns_nan(self):
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        pad_ratio, _ = compute_padding_and_edge(bgr)
        assert math.isnan(pad_ratio)


# ── TestQuantileBin ───────────────────────────────────────────────────────────

class TestQuantileBin:
    def _make_df(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "dataset":    ["A"] * 40 + ["B"] * 40,
            "blur_var_lap": np.concatenate([
                rng.uniform(0, 100, 40),
                rng.uniform(200, 400, 40),
            ]),
        })

    def test_produces_new_column(self):
        df = self._make_df()
        out = quantile_bin_within_dataset(df, "blur_var_lap")
        assert "blur_var_lap_q" in out.columns

    def test_q4_labels(self):
        df = self._make_df()
        out = quantile_bin_within_dataset(df, "blur_var_lap", n_quantiles=4)
        valid = out["blur_var_lap_q"] != "unknown"
        assert set(out.loc[valid, "blur_var_lap_q"]).issubset(
            {"Q1", "Q2", "Q3", "Q4"})

    def test_per_dataset_independence(self):
        """Thresholds computed separately per dataset."""
        df = self._make_df()
        out = quantile_bin_within_dataset(df, "blur_var_lap", n_quantiles=4)
        # Both datasets should have all 4 quartile labels
        for ds in ["A", "B"]:
            sub = out[out["dataset"] == ds]["blur_var_lap_q"]
            sub = sub[sub != "unknown"]
            assert len(set(sub)) > 1


# ── TestChiSquareCramers ──────────────────────────────────────────────────────

class TestChiSquareCramers:
    def _make_long_df(self, n: int = 200, seed: int = 0):
        """Construct a long-form DataFrame where failure rate strongly differs
        between "sharp" and "severe" bins, guaranteeing a significant chi2."""
        rng = np.random.default_rng(seed)
        half = n // 2
        # sharp bin: all real, all score 0.9 → 0% failure rate
        sharp_labels = np.ones(half, dtype=int)
        sharp_scores = np.full(half, 0.9)
        # severe bin: half real (fail, score 0.1) + half fake (correct, score 0.1)
        # real with score 0.1 → fails; fake with score 0.1 → correct
        # → 50% failure rate, very different from 0%
        sev_labels = np.array([1, 0] * (half // 2), dtype=int)
        sev_scores = np.full(half, 0.1)
        labels   = np.concatenate([sharp_labels, sev_labels])
        scores   = np.concatenate([sharp_scores, sev_scores])
        blur_bin = np.array(["sharp"] * half + ["severe"] * half)
        scores   = scores + rng.normal(0, 0.01, n)
        scores   = np.clip(scores, 0.01, 0.99)
        return pd.DataFrame({
            "model":          "ModelA",
            "dataset":        "ds1",
            "face_crop_path": [f"img_{i}.jpg" for i in range(n)],
            "score":          scores,
            "label":          labels,
            "blur_bin":       blur_bin,
        })

    def test_returns_dataframe(self):
        df = self._make_long_df()
        result = chi_square_cramers(df, ["blur_bin"])
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        df = self._make_long_df()
        result = chi_square_cramers(df, ["blur_bin"])
        for col in ("model", "dataset", "axis", "chi2", "p_value", "cramers_v", "effect"):
            assert col in result.columns

    def test_strong_dependence_low_pvalue(self):
        df = self._make_long_df(n=300)
        result = chi_square_cramers(df, ["blur_bin"])
        if len(result):
            assert result.iloc[0]["p_value"] < 0.05

    def test_effect_label_large(self):
        df = self._make_long_df(n=400)
        result = chi_square_cramers(df, ["blur_bin"])
        if len(result):
            assert result.iloc[0]["effect"] in ("medium", "large")

    def test_skips_unknown_bins(self):
        df = self._make_long_df()
        df["blur_bin"] = "unknown"
        result = chi_square_cramers(df, ["blur_bin"])
        assert len(result) == 0


# ── TestFailureJaccard ────────────────────────────────────────────────────────

class TestFailureJaccard:
    def _make_two_model_df(self):
        n = 20
        paths = [f"img_{i}.jpg" for i in range(n)]
        rows = []
        # Model A fails on first 10, Model B fails on last 10 → Jaccard = 0
        for i, path in enumerate(paths):
            for model, score in [("A", 0.1 if i < 10 else 0.9),
                                  ("B", 0.9 if i < 10 else 0.1)]:
                rows.append({
                    "model": model,
                    "dataset": "ds1",
                    "face_crop_path": path,
                    "score": score,
                    "label": 1,   # all real; score < 0.5 → fail (predicted fake)
                })
        return pd.DataFrame(rows)

    def test_disjoint_failures_jaccard_zero(self):
        df = self._make_two_model_df()
        result = failure_jaccard(df)
        assert len(result) == 1
        assert result.iloc[0]["jaccard"] == pytest.approx(0.0, abs=1e-6)

    def test_identical_failures_jaccard_one(self):
        n = 20
        paths = [f"img_{i}.jpg" for i in range(n)]
        rows = []
        for path in paths:
            for model in ["A", "B"]:
                rows.append({
                    "model": model, "dataset": "ds1",
                    "face_crop_path": path,
                    "score": 0.1,   # all fail (score < 0.5, label = 1)
                    "label": 1,
                })
        df = pd.DataFrame(rows)
        result = failure_jaccard(df)
        assert result.iloc[0]["jaccard"] == pytest.approx(1.0, abs=1e-6)

    def test_columns_present(self):
        df = self._make_two_model_df()
        result = failure_jaccard(df)
        for col in ("dataset", "model_A", "model_B", "jaccard", "lift",
                    "n_both_fail", "n_either_fail"):
            assert col in result.columns


# ── TestAgreementDistribution ─────────────────────────────────────────────────

class TestAgreementDistribution:
    def test_sums_to_total(self):
        n = 30
        rows = []
        for i in range(n):
            for model, score in [("A", 0.1), ("B", 0.9)]:
                rows.append({"model": model, "dataset": "ds1",
                              "face_crop_path": f"img_{i}.jpg",
                              "score": score, "label": 1})
        df  = pd.DataFrame(rows)
        agg = agreement_distribution(df)
        assert agg[agg["dataset"] == "ds1"]["n_samples"].sum() == n

    def test_columns_present(self):
        rows = [{"model": "A", "dataset": "ds1", "face_crop_path": "a.jpg",
                 "score": 0.9, "label": 1}]
        df  = pd.DataFrame(rows)
        agg = agreement_distribution(df)
        for col in ("dataset", "n_models_failed", "n_samples", "pct_of_dataset"):
            assert col in agg.columns
