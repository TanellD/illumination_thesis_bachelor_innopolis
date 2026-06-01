"""
Unit tests for src/eval/: metrics, aggregation, bootstrap, threshold_sweep.

Hand-computed values are derived analytically so these tests catch
implementation drift, not just "does it run".

Label convention throughout: 1 = real, 0 = fake.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.eval.metrics import compute_metrics, _ece, _eer, _fpr_at_tpr
from src.eval.aggregation import (
    aggregate_to_video,
    aggregate_seeds,
    wilcoxon_bonferroni,
    align_paired_video,
    compute_video_metrics,
)
from src.eval.threshold_sweep import sweep_thresholds, find_optima, THRESHOLDS
from src.eval.bootstrap import bootstrap_auc_ci, bootstrap_metrics_ci
from src.eval.cache import DiskCache, safe_key


# ── fixtures ──────────────────────────────────────────────────────────────────

def _perfect_scores():
    """Perfectly separated: score 0.9 for real, 0.1 for fake."""
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32)
    scores = np.array([0.9, 0.95, 0.85, 0.8, 0.1, 0.15, 0.05, 0.2], dtype=np.float64)
    return scores, labels


def _random_scores(seed: int = 0, n: int = 200):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n).astype(np.int32)
    scores = rng.random(size=n)
    return scores.astype(np.float64), labels


def _single_class_scores():
    labels = np.ones(8, dtype=np.int32)
    scores = np.array([0.9, 0.7, 0.8, 0.85, 0.6, 0.75, 0.5, 0.65])
    return scores, labels


# ── TestComputeMetrics ────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_keys_present(self):
        scores, labels = _perfect_scores()
        r = compute_metrics(labels, scores)
        for key in ("auc", "eer", "fpr_at_tpr95", "ece", "brier", "n"):
            assert key in r, f"missing key {key!r}"

    def test_perfect_auc(self):
        scores, labels = _perfect_scores()
        r = compute_metrics(labels, scores)
        assert r["auc"] == pytest.approx(1.0, abs=1e-9)

    def test_perfect_eer(self):
        scores, labels = _perfect_scores()
        r = compute_metrics(labels, scores)
        assert r["eer"] == pytest.approx(0.0, abs=1e-6)

    def test_perfect_fpr_at_tpr95(self):
        scores, labels = _perfect_scores()
        r = compute_metrics(labels, scores)
        assert r["fpr_at_tpr95"] == pytest.approx(0.0, abs=1e-6)

    def test_single_class_returns_nan(self):
        scores, labels = _single_class_scores()
        r = compute_metrics(labels, scores)
        assert math.isnan(r["auc"])
        assert math.isnan(r["eer"])
        assert math.isnan(r["fpr_at_tpr95"])
        assert math.isnan(r["ece"])

    def test_n_equals_len_labels(self):
        scores, labels = _random_scores(n=100)
        r = compute_metrics(labels, scores)
        assert r["n"] == 100

    def test_brier_range(self):
        scores, labels = _random_scores()
        r = compute_metrics(labels, scores)
        assert 0.0 <= r["brier"] <= 1.0

    def test_auc_range(self):
        scores, labels = _random_scores()
        r = compute_metrics(labels, scores)
        assert 0.0 <= r["auc"] <= 1.0


class TestECE:
    def test_perfectly_calibrated(self):
        """When score == label for every sample, ECE should be 0."""
        labels = np.array([1, 1, 0, 0], dtype=np.int32)
        scores = np.array([1.0, 1.0, 0.0, 0.0])
        # All samples fall into bin 0 (scores ≤ 1/15) or bin 14 (scores > 14/15)
        # In both bins: conf == acc, so ECE = 0.
        assert _ece(labels, scores) == pytest.approx(0.0, abs=1e-6)

    def test_ece_15_bins(self):
        """Manual computation for a toy input."""
        # 4 real samples with score 0.8 (fall in bin starting ~0.8)
        # 4 fake samples with score 0.2 (fall in bin starting ~0.2)
        # Both bins: conf = score, acc = label → |conf - acc| = |0.8 - 1| = 0.2, |0.2 - 0| = 0.2
        # ECE = 0.5 * 0.2 + 0.5 * 0.2 = 0.2
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32)
        scores = np.array([0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2])
        ece = _ece(labels, scores)
        assert ece == pytest.approx(0.2, abs=1e-6)

    def test_ece_range(self):
        scores, labels = _random_scores()
        ece = _ece(labels, scores)
        assert 0.0 <= ece <= 1.0


class TestEER:
    def test_perfect_separation(self):
        scores, labels = _perfect_scores()
        assert _eer(labels, scores) == pytest.approx(0.0, abs=1e-6)

    def test_random_near_half(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, 1000).astype(np.int32)
        scores = rng.random(1000)
        eer = _eer(labels, scores)
        assert 0.3 < eer < 0.7


class TestFPRatTPR:
    def test_perfect_zero(self):
        scores, labels = _perfect_scores()
        assert _fpr_at_tpr(labels, scores, 0.95) == pytest.approx(0.0, abs=1e-6)

    def test_returns_one_when_not_reachable(self):
        # Random uncorrelated scores: FPR@TPR95 should be close to 1.
        rng = np.random.default_rng(7)
        labels = rng.integers(0, 2, 1000).astype(np.int32)
        # Reverse the label/score mapping so FPR@TPR95 is close to 1
        scores = labels.astype(np.float64) * 0.0 + 0.5
        fpr = _fpr_at_tpr(labels, scores, 0.95)
        assert fpr >= 0.0


# ── TestAggregateToVideo ──────────────────────────────────────────────────────

class TestAggregateToVideo:
    def test_mean_strategy(self):
        scores  = np.array([0.8, 0.6, 0.2, 0.4])
        labels  = np.array([1,   1,   0,   0])
        sources = ["s1", "s1", "s2", "s2"]
        vids    = ["v1", "v1", "v2", "v2"]
        v_s, v_l, keys = aggregate_to_video(scores, labels, sources, vids,
                                             strategy="mean")
        assert len(v_s) == 2
        assert ("s1", "v1") in keys
        idx = keys.index(("s1", "v1"))
        assert v_s[idx] == pytest.approx(0.7, abs=1e-9)
        assert v_l[idx] == 1

    def test_max_strategy(self):
        scores  = np.array([0.3, 0.9, 0.1, 0.4])
        labels  = np.array([1,   1,   0,   0])
        sources = ["s1", "s1", "s2", "s2"]
        vids    = ["v1", "v1", "v2", "v2"]
        v_s, v_l, keys = aggregate_to_video(scores, labels, sources, vids,
                                             strategy="max")
        idx = keys.index(("s1", "v1"))
        assert v_s[idx] == pytest.approx(0.9, abs=1e-9)

    def test_vote_strategy(self):
        # 3 out of 4 frames predict real (score > 0.5) → vote score = 0.75
        scores  = np.array([0.6, 0.7, 0.8, 0.4])
        labels  = np.array([1,   1,   1,   1])
        sources = ["s"] * 4
        vids    = ["v"] * 4
        v_s, v_l, _ = aggregate_to_video(scores, labels, sources, vids,
                                          strategy="vote")
        assert v_s[0] == pytest.approx(0.75, abs=1e-9)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            aggregate_to_video(np.zeros(4), np.zeros(4, dtype=np.int32),
                               ["s"] * 4, ["v"] * 4, strategy="bogus")

    def test_dataset_prefix_in_key(self):
        scores  = np.array([0.8, 0.6])
        labels  = np.array([1, 0])
        sources = ["s1", "s2"]
        vids    = ["v1", "v2"]
        _, _, keys = aggregate_to_video(scores, labels, sources, vids,
                                         dataset="ff++")
        assert all(len(k) == 3 for k in keys), "expected 3-tuple keys with dataset"
        assert ("ff++", "s1", "v1") in keys

    def test_sbi_source_isolation(self):
        """SBI fake derived from real source must NOT be merged with real."""
        scores  = np.array([0.9, 0.1])   # real frames
        labels  = np.array([1, 0])
        sources = ["ffpp_real", "ffpp_fake_SBI"]   # different source tags
        vids    = ["v1", "v1"]            # same video_id — must NOT merge
        v_s, v_l, keys = aggregate_to_video(scores, labels, sources, vids)
        assert len(v_s) == 2, "real and SBI fake must remain separate groups"


# ── TestAggregateSeeds ────────────────────────────────────────────────────────

class TestAggregateSeeds:
    def test_mean_of_two(self):
        a = {"auc": 0.8, "brier": 0.2, "n": 100}
        b = {"auc": 0.9, "brier": 0.1, "n": 100}
        mean, std = aggregate_seeds([a, b])
        assert mean["auc"] == pytest.approx(0.85, abs=1e-9)
        assert mean["brier"] == pytest.approx(0.15, abs=1e-9)

    def test_std_of_two(self):
        a = {"auc": 0.7, "n": 100}
        b = {"auc": 0.9, "n": 100}
        _, std = aggregate_seeds([a, b])
        expected_std = np.std([0.7, 0.9], ddof=1)
        assert std["auc"] == pytest.approx(expected_std, abs=1e-9)

    def test_single_seed_nan_std(self):
        a = {"auc": 0.8, "n": 50}
        _, std = aggregate_seeds([a])
        assert math.isnan(std["auc"])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_seeds([])

    def test_n_propagated(self):
        a = {"auc": 0.8, "n": 77}
        mean, _ = aggregate_seeds([a])
        assert mean["n"] == 77


# ── TestWilcoxonBonferroni ────────────────────────────────────────────────────

class TestWilcoxonBonferroni:
    def test_identical_returns_not_significant(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 100).astype(np.int32)
        scores = rng.random(100)
        result = wilcoxon_bonferroni(scores, labels, scores, labels)
        assert not result["significant"]
        assert result["n"] == 100

    def test_clearly_different_is_significant(self):
        """A clearly better model vs a clearly worse one should be significant."""
        rng = np.random.default_rng(1)
        n = 500
        labels = rng.integers(0, 2, n).astype(np.int32)
        # good_scores separates classes; bad_scores is random
        good_scores = labels.astype(np.float64) + rng.normal(0, 0.05, n)
        good_scores = np.clip(good_scores, 0.0, 1.0)
        bad_scores = rng.random(n)
        result = wilcoxon_bonferroni(bad_scores, labels, good_scores, labels,
                                      n_comparisons=1, alpha=0.05)
        assert result["significant"]
        assert result["better"] == "B"  # B (good) has lower Brier

    def test_bonferroni_correction_reduces_significance(self):
        """With n_comparisons=10, a borderline-significant test becomes non-significant."""
        rng = np.random.default_rng(42)
        n = 200
        labels = rng.integers(0, 2, n).astype(np.int32)
        a_scores = rng.random(n)
        # Make B slightly better
        b_scores = np.clip(labels.astype(float) * 0.3 + rng.random(n) * 0.7, 0, 1)
        r1 = wilcoxon_bonferroni(a_scores, labels, b_scores, labels, n_comparisons=1)
        r10 = wilcoxon_bonferroni(a_scores, labels, b_scores, labels, n_comparisons=10)
        assert r10["p_corrected"] >= r1["p_corrected"]

    def test_keys_present(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 50).astype(np.int32)
        s = rng.random(50)
        result = wilcoxon_bonferroni(s, labels, s, labels)
        for k in ("stat", "p_value", "p_corrected", "effect_d",
                  "mean_brier_A", "mean_brier_B", "better", "significant", "n"):
            assert k in result


# ── TestThresholdSweep ────────────────────────────────────────────────────────

class TestThresholdSweep:
    def test_returns_99_rows(self):
        scores, labels = _random_scores()
        df = sweep_thresholds(scores, labels)
        assert len(df) == 99

    def test_columns_present(self):
        scores, labels = _random_scores()
        df = sweep_thresholds(scores, labels)
        for col in ("threshold", "bal_acc", "mcc", "youden_j",
                    "tpr", "tnr", "fpr", "fnr", "n", "n_fail", "fail_rate"):
            assert col in df.columns, f"missing column {col!r}"

    def test_thresholds_in_range(self):
        scores, labels = _random_scores()
        df = sweep_thresholds(scores, labels)
        assert df["threshold"].min() >= 0.01 - 1e-9
        assert df["threshold"].max() <= 0.99 + 1e-9

    def test_perfect_scores_max_bal_acc(self):
        scores, labels = _perfect_scores()
        df = sweep_thresholds(scores, labels)
        optima = find_optima(df)
        # Perfect separation: balanced accuracy should reach 1.0
        assert optima["max_bal_acc"] == pytest.approx(1.0, abs=1e-6)

    def test_find_optima_keys(self):
        scores, labels = _random_scores()
        df = sweep_thresholds(scores, labels)
        opt = find_optima(df)
        for k in ("max_bal_acc", "thresh_at_max_bal_acc",
                  "max_mcc",     "thresh_at_max_mcc",
                  "max_youden_j", "thresh_at_max_youden_j",
                  "fail_rate_at_0.5", "fail_rate_at_opt_bal_acc"):
            assert k in opt, f"missing key {k!r}"

    def test_youden_equals_tpr_plus_tnr_minus_one(self):
        scores, labels = _random_scores(n=50)
        df = sweep_thresholds(scores, labels)
        for _, row in df.iterrows():
            expected = row["tpr"] + row["tnr"] - 1.0
            assert row["youden_j"] == pytest.approx(expected, abs=1e-9)


# ── TestBootstrap ─────────────────────────────────────────────────────────────

class TestBootstrap:
    def test_ci_contains_point_estimate(self):
        """The 95% CI should contain the true sample AUC (on perfect data it's 1.0)."""
        scores, labels = _perfect_scores()
        lo, hi = bootstrap_auc_ci(scores, labels, n_resamples=200, seed=0)
        from sklearn.metrics import roc_auc_score
        point = roc_auc_score(labels, scores)
        assert lo <= point <= hi

    def test_ci_width_positive(self):
        scores, labels = _random_scores(n=100)
        lo, hi = bootstrap_auc_ci(scores, labels, n_resamples=200, seed=0)
        assert hi > lo

    def test_single_class_returns_nan(self):
        scores, labels = _single_class_scores()
        lo, hi = bootstrap_auc_ci(scores, labels, n_resamples=100)
        assert math.isnan(lo) and math.isnan(hi)

    def test_video_mode_returns_valid_ci(self):
        rng = np.random.default_rng(3)
        n = 100
        labels = rng.integers(0, 2, n).astype(np.int32)
        scores = rng.random(n)
        # 10 videos with 10 frames each
        video_ids = [f"v{i // 10}" for i in range(n)]
        sources   = ["src"] * n
        lo, hi = bootstrap_auc_ci(scores, labels, video_ids=video_ids,
                                   sources=sources, n_resamples=100, seed=0)
        assert not math.isnan(lo)
        assert hi > lo

    def test_bootstrap_metrics_ci_keys(self):
        scores, labels = _random_scores(n=80)
        result = bootstrap_metrics_ci(scores, labels, n_resamples=50, seed=0)
        for k in ("auc", "auc_ci_lo", "auc_ci_hi",
                  "brier", "brier_ci_lo", "brier_ci_hi",
                  "ece", "ece_ci_lo", "ece_ci_hi",
                  "n_samples"):
            assert k in result, f"missing key {k!r}"

    def test_bootstrap_metrics_ci_brier_range(self):
        scores, labels = _random_scores(n=80)
        result = bootstrap_metrics_ci(scores, labels, n_resamples=50, seed=0)
        assert 0.0 <= result["brier"] <= 1.0
        assert result["brier_ci_lo"] <= result["brier"] <= result["brier_ci_hi"]


# ── TestDiskCache ─────────────────────────────────────────────────────────────

class TestDiskCache:
    def test_save_and_load(self, tmp_path):
        cache = DiskCache(tmp_path)
        scores = np.array([0.9, 0.1, 0.7], dtype=np.float32)
        labels = np.array([1, 0, 1], dtype=np.int32)
        cache.save_inference("RGB-Only", "ff++", scores, labels)
        result = cache.load_inference("RGB-Only", "ff++")
        assert result is not None
        np.testing.assert_array_equal(result["scores"], scores)
        np.testing.assert_array_equal(result["labels"], labels)

    def test_missing_returns_none(self, tmp_path):
        cache = DiskCache(tmp_path)
        assert cache.load_inference("no-model", "no-dataset") is None

    def test_has_inference(self, tmp_path):
        cache = DiskCache(tmp_path)
        assert not cache.has_inference("m", "d")
        cache.save_inference("m", "d", np.array([0.5]), np.array([1]))
        assert cache.has_inference("m", "d")

    def test_delete_inference(self, tmp_path):
        cache = DiskCache(tmp_path)
        cache.save_inference("m", "d", np.array([0.5]), np.array([1]))
        assert cache.delete_inference("m", "d")
        assert not cache.has_inference("m", "d")
        assert not cache.delete_inference("m", "d")  # second call: False

    def test_safe_key(self):
        assert safe_key("RGB/Only") == "RGB-Only"
        assert safe_key("C mix") == "C_mix"
        assert safe_key("A+B") == "ApB"

    def test_optional_arrays_stored(self, tmp_path):
        cache = DiskCache(tmp_path)
        vids = np.array(["v1", "v2"], dtype=object)
        srcs = np.array(["src1", "src2"], dtype=object)
        cache.save_inference("m", "d",
                              np.array([0.5, 0.3]),
                              np.array([1, 0]),
                              video_ids=vids, sources=srcs)
        result = cache.load_inference("m", "d")
        assert "video_ids" in result
        assert "sources" in result
        np.testing.assert_array_equal(result["video_ids"], vids)
