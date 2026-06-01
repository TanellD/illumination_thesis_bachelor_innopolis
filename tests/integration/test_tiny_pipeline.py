"""
tests/integration/test_tiny_pipeline.py
========================================
End-to-end smoke test of the entire pipeline on the tiny-corpus fixture.

What it covers
--------------
Every pipeline stage is exercised in dependency order:

  1. Tiny corpus fixture exists and is valid (manifest + JPEG crops)
  2. src/data/manifest.py  — validate_manifest passes on fixture
  3. src/eval/metrics.py   — compute_metrics on fixture labels/scores
  4. src/eval/aggregation.py — aggregate_to_video on fixture data
  5. src/eval/bootstrap.py — bootstrap_auc_ci on fixture data
  6. src/eval/threshold_sweep.py — sweep_thresholds on fixture data
  7. src/eval/cache.py     — DiskCache save/load round-trip
  8. src/models/noiseprintpp.py — NoiseprintPlusPlus forward pass (random init)
  9. src/models/residual_only.py — ResidualOnlyModel CPU forward pass
  10. src/robustness/perturbations.py — all 5 families on a tiny batch
  11. src/taxonomy/attributes.py — compute_attributes on a fixture crop
  12. src/taxonomy/contingency.py — chi_square_cramers + failure_jaccard
  13. src/report/tables.py  — build_T1 with synthetic prediction CSV
  14. src/report/manifest_doc.py — write_results_md
  15. src/tsbi/generator.py — boxes_compatible, select_pair, t_histmatch
  16. src/tsbi/gating.py    — gate_pair + split_by_dl_quartile

Run: pytest tests/integration/test_tiny_pipeline.py -v
Expected: all pass in < 60 s on CPU (no GPU, no real data)

Integration contract
--------------------
This test must remain green after any refactor.  If a function name
changes, update the import here.  The test is deliberately not a unit
test — it exercises the actual import chain across module boundaries.
"""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# ── locate the tiny corpus fixture ───────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "tiny_corpus"
MANIFEST    = FIXTURE_DIR / "manifest.csv"


def _require_tiny_corpus():
    if not MANIFEST.exists():
        pytest.skip(
            "Tiny corpus not built. Run: "
            "python tests/fixtures/build_tiny_corpus.py "
            "--out tests/fixtures/tiny_corpus/"
        )


# ── 1. Fixture validity ───────────────────────────────────────────────────────

class TestTinyCorpusFixture:
    def test_manifest_exists(self):
        _require_tiny_corpus()
        assert MANIFEST.exists()

    def test_manifest_has_16_rows(self):
        _require_tiny_corpus()
        df = pd.read_csv(MANIFEST)
        assert len(df) == 16

    def test_manifest_has_required_columns(self):
        _require_tiny_corpus()
        from src.data.manifest import MANIFEST_COLUMNS
        df = pd.read_csv(MANIFEST)
        for col in MANIFEST_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_crop_files_exist(self):
        _require_tiny_corpus()
        df = pd.read_csv(MANIFEST)
        for path in df["face_crop_path"]:
            assert os.path.isfile(path), f"Crop file missing: {path}"

    def test_label_distribution(self):
        _require_tiny_corpus()
        df = pd.read_csv(MANIFEST)
        assert (df["label"] == 0).sum() == 8   # 8 real
        assert (df["label"] == 1).sum() == 8   # 8 fake

    def test_splits_present(self):
        _require_tiny_corpus()
        df = pd.read_csv(MANIFEST)
        assert set(df["split"].unique()) == {"train", "val", "test"}


# ── 2. Manifest validation ────────────────────────────────────────────────────

class TestManifestValidation:
    def test_validate_passes_on_fixture(self):
        _require_tiny_corpus()
        from src.data.manifest import validate_manifest
        df = pd.read_csv(MANIFEST)
        validate_manifest(df, check_files_exist=True)   # must not raise


# ── 3-6. Evaluation primitives ────────────────────────────────────────────────

class TestEvalPrimitives:
    @pytest.fixture(scope="class")
    def scores_labels(self):
        rng = np.random.default_rng(0)
        n = 50
        labels = rng.integers(0, 2, n).astype(np.int32)
        scores = np.clip(labels + rng.normal(0, 0.3, n), 0.0, 1.0)
        return scores, labels

    def test_compute_metrics_runs(self, scores_labels):
        from src.eval.metrics import compute_metrics
        scores, labels = scores_labels
        m = compute_metrics(labels, scores)
        assert 0.0 <= m["auc"] <= 1.0
        assert 0.0 <= m["ece"] <= 1.0

    def test_aggregate_to_video(self, scores_labels):
        from src.eval.aggregation import aggregate_to_video
        scores, labels = scores_labels
        n = len(scores)
        sources  = [f"s{i % 5}" for i in range(n)]
        vids     = [f"v{i // 2}" for i in range(n)]
        v_s, v_l, keys = aggregate_to_video(scores, labels, sources, vids)
        assert len(v_s) > 0
        assert len(v_s) == len(v_l) == len(keys)

    def test_bootstrap_auc_ci(self, scores_labels):
        from src.eval.bootstrap import bootstrap_auc_ci
        scores, labels = scores_labels
        lo, hi = bootstrap_auc_ci(scores, labels, n_resamples=50, seed=0)
        assert lo <= hi

    def test_threshold_sweep(self, scores_labels):
        from src.eval.threshold_sweep import sweep_thresholds, find_optima
        scores, labels = scores_labels
        df = sweep_thresholds(scores, labels)
        assert len(df) == 99
        opt = find_optima(df)
        assert "max_bal_acc" in opt


# ── 7. DiskCache ──────────────────────────────────────────────────────────────

class TestDiskCache:
    def test_roundtrip(self, tmp_path):
        from src.eval.cache import DiskCache
        cache = DiskCache(tmp_path)
        s = np.array([0.9, 0.1], dtype=np.float32)
        l = np.array([1, 0],     dtype=np.int32)
        cache.save_inference("M", "D", s, l)
        result = cache.load_inference("M", "D")
        assert result is not None
        np.testing.assert_array_equal(result["scores"], s)


# ── 8. NoiseprintPlusPlus (random init, no weights file) ─────────────────────

class TestNoiseprintForward:
    def test_forward_pass_shape(self):
        from src.models.noiseprintpp import NoiseprintPlusPlus
        model = NoiseprintPlusPlus()
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 32, 32)


# ── 9. ResidualOnlyModel CPU forward ─────────────────────────────────────────

class TestResidualOnlyForward:
    def test_forward_no_noise_model(self):
        from src.models.residual_only import ResidualOnlyModel
        model = ResidualOnlyModel(num_classes=2)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)

    def test_instance_norm_is_present(self):
        import torch.nn as nn
        from src.models.residual_only import ResidualOnlyModel
        model = ResidualOnlyModel()
        assert isinstance(model.noise_norm, nn.InstanceNorm2d)


# ── 10. Robustness perturbations ──────────────────────────────────────────────

class TestPerturbationsIntegration:
    def test_all_families_run_on_small_batch(self):
        from src.robustness.perturbations import PERTURBATION_GRID, apply_perturbation
        batch = torch.rand(1, 3, 16, 16) * 2 - 1   # [-1, 1] normalised
        for family, param in PERTURBATION_GRID:
            out = apply_perturbation(batch, family, param)
            assert out.shape == batch.shape, f"Shape changed for ({family}, {param})"


# ── 11. Taxonomy attribute extraction ────────────────────────────────────────

class TestAttributeExtraction:
    def test_compute_attributes_on_fixture(self):
        _require_tiny_corpus()
        from src.taxonomy.attributes import compute_attributes
        df = pd.read_csv(MANIFEST)
        path = df["face_crop_path"].iloc[0]
        attrs = compute_attributes(path)
        assert attrs.decoded_ok
        assert attrs.image_h > 0
        assert not np.isnan(attrs.blur_var_lap)
        assert attrs.blur_bin in ("severe", "moderate", "mild", "sharp")

    def test_batch_attribute_extraction(self, tmp_path):
        _require_tiny_corpus()
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not installed — skipping parquet write test")
        from src.taxonomy.attributes import compute_attributes_batch
        df    = pd.read_csv(MANIFEST)
        paths = df["face_crop_path"].tolist()
        out_parquet = str(tmp_path / "attrs.parquet")
        result_df = compute_attributes_batch(
            paths=paths[:4],
            output_parquet=out_parquet,
            checkpoint_every=0,
            resume=False,
        )
        assert len(result_df) == 4
        assert os.path.exists(out_parquet)


# ── 12. Contingency tests ─────────────────────────────────────────────────────

class TestContingencyIntegration:
    def _make_long_df(self):
        """Deterministic long-form DataFrame with real+fake, two blur bins."""
        n = 80
        half = n // 2
        labels   = np.array([1] * half + [0] * (half // 2) + [1] * (half // 2))
        scores   = np.concatenate([
            np.full(half, 0.9),      # all real, predict real correctly
            np.full(half // 2, 0.9), # fake predicted real (fail)
            np.full(half // 2, 0.1), # real predicted fake (fail)
        ])
        blur_bin = ["sharp"] * half + ["severe"] * half
        return pd.DataFrame({
            "model":          "M",
            "dataset":        "D",
            "face_crop_path": [f"img_{i}.jpg" for i in range(n)],
            "score":          scores,
            "label":          labels.astype(int),
            "blur_bin":       blur_bin,
        })

    def test_chi_square_returns_df(self):
        from src.taxonomy.contingency import chi_square_cramers
        df = self._make_long_df()
        result = chi_square_cramers(df, ["blur_bin"])
        assert isinstance(result, pd.DataFrame)

    def test_jaccard_returns_df(self):
        from src.taxonomy.contingency import failure_jaccard
        long_df = self._make_long_df()
        # Add a second model
        df2 = long_df.copy()
        df2["model"] = "M2"
        combined = pd.concat([long_df, df2], ignore_index=True)
        result = failure_jaccard(combined)
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "jaccard" in result.columns


# ── 13. Table builder (T1) with synthetic predictions ────────────────────────

class TestT1WithSyntheticPreds:
    def _make_pred_csv(self, tmp_dir: str, regime: str, mkey: str) -> None:
        rng = np.random.default_rng(0)
        n = 40
        labels  = rng.integers(0, 2, n).astype(int)
        scores  = np.clip(labels + rng.normal(0, 0.2, n), 0.0, 1.0)
        df = pd.DataFrame({
            "score":    scores,
            "label":    labels,
            "source":   [f"s{i % 4}" for i in range(n)],
            "video_id": [f"v{i // 4}" for i in range(n)],
            "method":   ["Deepfakes"] * n,
            "path":     [f"/p/{i}.jpg" for i in range(n)],
        })
        df.to_csv(os.path.join(tmp_dir, f"{regime}__{mkey}.csv"), index=False)

    def test_t1_produces_dataframe_and_latex(self, tmp_path):
        from src.report.tables import build_T1
        preds_dir = str(tmp_path)
        self._make_pred_csv(preds_dir, "RGB-Only", "FF++_stage1")
        cfg = {"table_assignments": {"T1": ["RGB-Only"]}}
        tidy, latex = build_T1(preds_dir, cfg)
        assert len(tidy) == 1
        assert 0.0 <= tidy.iloc[0]["auc"] <= 1.0
        assert "\\begin{table}" in latex


# ── 14. RESULTS.md assembly ───────────────────────────────────────────────────

class TestResultsMdAssembly:
    def test_write_results_md(self, tmp_path):
        from src.report.manifest_doc import write_results_md
        run_info = {
            "git_sha":     "abc123",
            "config_hash": "xyz456",
            "tables":  {"T1": {"csv": "/t/T1.csv", "tex": "/t/T1.tex"}},
            "figures": {"reliability": "/f/rel.png"},
            "experiments": [{"name": "b1", "seeds": [42],
                              "elapsed_s": 10, "output_dir": "/o/b1"}],
        }
        path = write_results_md(str(tmp_path), run_info)
        assert os.path.exists(path)
        content = open(path).read()
        assert "abc123" in content
        assert "T1" in content
        assert "reliability" in content
        assert "Reproduction" in content


# ── 15. T-SBI pair geometry ───────────────────────────────────────────────────

class TestTSBIGeometry:
    def test_boxes_compatible_identical(self):
        from src.tsbi.generator import boxes_compatible
        b = (100, 50, 60, 70)
        assert boxes_compatible(b, b) is True

    def test_boxes_too_large_scale(self):
        from src.tsbi.generator import boxes_compatible
        a = (0, 0, 60, 60)
        b = (0, 0, 90, 90)   # 1.5x scale
        assert boxes_compatible(a, b, max_scale=1.3) is False

    def test_histmatch_on_synthetic_frame(self):
        from src.tsbi.generator import t_histmatch
        rng = np.random.default_rng(1)
        tgt  = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        src  = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 200, dtype=np.uint8)
        out  = t_histmatch(tgt, src, mask)
        assert out.shape == (64, 64, 3)
        assert out.dtype.itemsize == 1   # uint8


# ── 16. Illumination gating ───────────────────────────────────────────────────

class TestIlluminationGating:
    def test_gate_pair_thresholds(self):
        from src.tsbi.gating import gate_pair, MIN_ILLUM_DELTA_MEAN, MIN_ILLUM_DELTA_STD
        assert gate_pair(MIN_ILLUM_DELTA_MEAN, 0.0) is True
        assert gate_pair(0.0, MIN_ILLUM_DELTA_STD) is True
        assert gate_pair(MIN_ILLUM_DELTA_MEAN - 1, MIN_ILLUM_DELTA_STD - 1) is False

    def test_dl_quartile_split_smoke(self):
        from src.tsbi.gating import split_by_dl_quartile
        rng = np.random.default_rng(2)
        n   = 60
        df  = pd.DataFrame({
            "face_crop_path": [f"/p{i}.jpg" for i in range(n)],
            "label":          rng.integers(0, 2, n),
            "source":         ["tsbi"] * 40 + ["sbi"] * 20,
            "split":          "train",
        })
        dl = {f"/p{i}.jpg": float(i) for i in range(40)}
        high, low, hi_thr, lo_thr = split_by_dl_quartile(
            df, dl, compute_thresholds=True)
        assert hi_thr > lo_thr
        assert len(high) > 0 and len(low) > 0
