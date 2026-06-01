"""
Unit tests for src/report/ — table builders and RESULTS.md assembler.

Tests verify:
- Every table builder returns (DataFrame, str) even when prediction CSVs are missing
- Missing prediction files produce "--" cells in LaTeX, NaN in tidy CSV
- Table tidy DataFrames have the correct columns
- LaTeX output contains \\begin{table} and the correct label
- build_all_tables correctly routes and writes files
- write_results_md produces a well-formed markdown file with correct sections
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.report.tables import (
    build_T1,
    build_T2,
    build_T3,
    build_T6,
    build_T7,
    build_T10,
    build_all_tables,
    _latex_table,
    _fmt,
)
from src.report.manifest_doc import write_results_md


# ── synthetic prediction CSV fixture ─────────────────────────────────────────

def _make_preds_csv(tmp_dir: str, regime: str, mkey: str,
                    ptag: str = "", n: int = 100, seed: int = 0) -> str:
    """Write a synthetic prediction CSV and return its path."""
    rng = np.random.default_rng(seed)
    labels  = rng.integers(0, 2, n).astype(int)
    scores  = np.clip(labels.astype(float) + rng.normal(0, 0.3, n), 0.0, 1.0)
    sources = [f"src_{i % 5}" for i in range(n)]
    vids    = [f"v{i // 5}" for i in range(n)]
    methods = ["Deepfakes"] * (n // 2) + ["Face2Face"] * (n - n // 2)
    df = pd.DataFrame({
        "score":    scores,
        "label":    labels,
        "source":   sources,
        "video_id": vids,
        "method":   methods,
        "path":     [f"/path/img_{i}.jpg" for i in range(n)],
    })
    fname = f"{regime}__{mkey}"
    if ptag:
        fname += f"__{ptag}"
    path = os.path.join(tmp_dir, fname + ".csv")
    df.to_csv(path, index=False)
    return path


# ── TestFmt ────────────────────────────────────────────────────────────────────

class TestFmt:
    def test_float_3dp(self):
        assert _fmt(0.1234) == "0.123"

    def test_nan_returns_dash(self):
        assert _fmt(float("nan")) == "--"

    def test_inf_returns_dash(self):
        assert _fmt(float("inf")) == "--"

    def test_none_returns_dash(self):
        assert _fmt(None) == "--"


# ── TestLatexTable ─────────────────────────────────────────────────────────────

class TestLatexTable:
    def test_contains_begin_table(self):
        result = _latex_table(["A", "B"], [["1", "2"]])
        assert "\\begin{table}" in result

    def test_contains_end_table(self):
        result = _latex_table(["A", "B"], [["1", "2"]])
        assert "\\end{table}" in result

    def test_caption_included(self):
        result = _latex_table(["H"], [["x"]], caption="My caption")
        assert "My caption" in result

    def test_label_included(self):
        result = _latex_table(["H"], [["x"]], label="tab:test")
        assert "tab:test" in result

    def test_headers_bolded(self):
        result = _latex_table(["ModelCol"], [["val"]])
        assert "\\textbf{ModelCol}" in result


# ── TestBuildT1 ────────────────────────────────────────────────────────────────

class TestBuildT1:
    def _make_cfg(self, models):
        return {"table_assignments": {"T1": models}}

    def test_missing_preds_returns_nan(self, tmp_path):
        cfg = self._make_cfg(["RGB-Only"])
        tidy, latex = build_T1(str(tmp_path), cfg)
        assert len(tidy) == 1
        assert np.isnan(tidy.iloc[0]["auc"])
        assert "--" in latex

    def test_with_predictions_returns_finite_auc(self, tmp_path):
        _make_preds_csv(str(tmp_path), "RGB-Only", "FF++_stage1")
        cfg = self._make_cfg(["RGB-Only"])
        tidy, latex = build_T1(str(tmp_path), cfg)
        auc = tidy.iloc[0]["auc"]
        assert 0.0 <= auc <= 1.0

    def test_columns_present(self, tmp_path):
        cfg = self._make_cfg(["M"])
        tidy, _ = build_T1(str(tmp_path), cfg)
        for col in ("model", "auc", "eer", "fpr_at_tpr95", "brier", "ece"):
            assert col in tidy.columns

    def test_latex_has_correct_label(self, tmp_path):
        cfg = self._make_cfg(["M"])
        _, latex = build_T1(str(tmp_path), cfg)
        assert "tab:t1_stage1_ffpp" in latex


# ── TestBuildT2 ────────────────────────────────────────────────────────────────

class TestBuildT2:
    def _make_cfg(self, baselines):
        return {"table_assignments": {"T2": {
            "baselines": baselines,
            "fixed": {},
        }}}

    def test_missing_returns_nan(self, tmp_path):
        cfg = self._make_cfg(["RGB-Only"])
        tidy, _ = build_T2(str(tmp_path), cfg)
        assert len(tidy) == 1
        assert np.isnan(tidy.iloc[0]["ff_auc"])

    def test_columns_present(self, tmp_path):
        cfg = self._make_cfg(["RGB-Only"])
        tidy, _ = build_T2(str(tmp_path), cfg)
        for col in ("model", "ff_auc", "cdf_auc", "seeds"):
            assert col in tidy.columns

    def test_baseline_row_has_zero_delta(self, tmp_path):
        _make_preds_csv(str(tmp_path), "RGB-Only", "FF++_stage1")
        _make_preds_csv(str(tmp_path), "RGB-Only", "CelebDF")
        cfg = self._make_cfg(["RGB-Only"])
        tidy, _ = build_T2(str(tmp_path), cfg)
        assert tidy.iloc[0]["ff_delta"] == pytest.approx(0.0, abs=1e-9)


# ── TestBuildT3 ────────────────────────────────────────────────────────────────

class TestBuildT3:
    def _make_cfg(self, models):
        return {"table_assignments": {"T3": models}}

    def test_missing_all_returns_nan(self, tmp_path):
        cfg = self._make_cfg(["M1"])
        tidy, _ = build_T3(str(tmp_path), cfg)
        assert np.isnan(tidy.iloc[0]["FF++"])

    def test_columns_include_all_datasets(self, tmp_path):
        cfg = self._make_cfg(["M1"])
        tidy, _ = build_T3(str(tmp_path), cfg)
        for col in ("model", "FF++", "Celeb-DF", "DFDC", "DFF"):
            assert col in tidy.columns

    def test_with_predictions_finite(self, tmp_path):
        _make_preds_csv(str(tmp_path), "M1", "FF++_stage1")
        cfg = self._make_cfg(["M1"])
        tidy, _ = build_T3(str(tmp_path), cfg)
        assert 0.0 <= tidy.iloc[0]["FF++"] <= 1.0


# ── TestBuildT6 ────────────────────────────────────────────────────────────────

class TestBuildT6:
    def _make_cfg(self, regimes):
        return {"table_assignments": {"T6": regimes}}

    def test_missing_returns_nan(self, tmp_path):
        cfg = self._make_cfg(["A", "B_mix"])
        tidy, _ = build_T6(str(tmp_path), cfg)
        assert len(tidy) == 2
        assert np.isnan(tidy.iloc[0]["Celeb-DF"])

    def test_columns_present(self, tmp_path):
        cfg = self._make_cfg(["A"])
        tidy, _ = build_T6(str(tmp_path), cfg)
        for col in ("regime", "Celeb-DF", "DFDC", "DFF"):
            assert col in tidy.columns


# ── TestBuildT7 ────────────────────────────────────────────────────────────────

class TestBuildT7:
    def _make_cfg(self, regimes):
        return {"table_assignments": {"T7": regimes}}

    def test_columns_present(self, tmp_path):
        cfg = self._make_cfg(["A"])
        tidy, _ = build_T7(str(tmp_path), cfg)
        for col in ("regime", "celebdf_ece", "dfdc_ece",
                    "celebdf_brier", "dfdc_brier"):
            assert col in tidy.columns


# ── TestBuildT10 ───────────────────────────────────────────────────────────────

class TestBuildT10:
    def test_missing_csv_returns_empty_df(self, tmp_path):
        cfg = {"table_assignments": {"T10": {"tsbi_labels_csv": "/nonexistent.csv"}}}
        tidy, latex = build_T10(str(tmp_path), cfg)
        assert "stat" in tidy.columns
        assert "pending" in latex.lower() or "(tsbi" in latex.lower()

    def test_with_tsbi_csv(self, tmp_path):
        tsbi_csv = str(tmp_path / "tsbi_labels.csv")
        pd.DataFrame({
            "face_crop_path": [f"/p{i}.jpg" for i in range(20)],
            "illum_delta_L":   np.random.uniform(0, 20, 20),
            "illum_delta_Lstd":np.random.uniform(0, 10, 20),
            "illum_relaxed":   np.zeros(20),
        }).to_csv(tsbi_csv, index=False)
        cfg = {"table_assignments": {"T10": {"tsbi_labels_csv": tsbi_csv}}}
        tidy, latex = build_T10(str(tmp_path), cfg)
        assert len(tidy) > 0
        assert "\\begin{table}" in latex


# ── TestBuildAllTables ─────────────────────────────────────────────────────────

class TestBuildAllTables:
    def test_writes_csv_and_tex(self, tmp_path):
        preds_dir = str(tmp_path / "preds")
        os.makedirs(preds_dir, exist_ok=True)
        _make_preds_csv(preds_dir, "M1", "FF++_stage1")

        cfg = {"table_assignments": {"T1": ["M1"]}}
        results = build_all_tables(preds_dir, cfg, str(tmp_path), tables=["T1"])
        assert "T1" in results
        assert os.path.exists(tmp_path / "tables_csv" / "T1.csv")
        assert os.path.exists(tmp_path / "tables_tex" / "T1.tex")

    def test_skips_tables_missing_from_assignments(self, tmp_path):
        cfg = {"table_assignments": {"T1": ["M1"]}}
        results = build_all_tables(str(tmp_path), cfg, str(tmp_path),
                                    tables=["T1", "T3"])
        # T3 is not in table_assignments → should be skipped
        assert "T3" not in results


# ── TestWriteResultsMd ─────────────────────────────────────────────────────────

class TestWriteResultsMd:
    def _make_run_info(self):
        return {
            "config_hash":    "abc123",
            "git_sha":        "deadbeef",
            "timestamp":      "2026-01-01T00:00:00",
            "python_version": "3.10.0",
            "torch_version":  "2.2.2",
            "hostname":       "test-host",
            "gpu":            "A100",
            "tables":  {"T1": {"csv": "/tmp/T1.csv", "tex": "/tmp/T1.tex"}},
            "figures": {"reliability": "/tmp/reliability.png"},
            "experiments": [
                {"name": "b1_ablation", "seeds": [42, 123],
                 "elapsed_s": 300, "output_dir": "/out/b1"}
            ],
        }

    def test_creates_file(self, tmp_path):
        run_info = self._make_run_info()
        path = write_results_md(str(tmp_path), run_info)
        assert os.path.exists(path)
        assert path.endswith("RESULTS.md")

    def test_contains_git_sha(self, tmp_path):
        run_info = self._make_run_info()
        path = write_results_md(str(tmp_path), run_info)
        content = open(path).read()
        assert "deadbeef" in content

    def test_contains_table_section(self, tmp_path):
        run_info = self._make_run_info()
        path = write_results_md(str(tmp_path), run_info)
        content = open(path).read()
        assert "Tables" in content
        assert "T1" in content

    def test_contains_figures_section(self, tmp_path):
        run_info = self._make_run_info()
        path = write_results_md(str(tmp_path), run_info)
        content = open(path).read()
        assert "Figures" in content
        assert "reliability" in content

    def test_contains_experiments_section(self, tmp_path):
        run_info = self._make_run_info()
        path = write_results_md(str(tmp_path), run_info)
        content = open(path).read()
        assert "b1_ablation" in content

    def test_contains_reproduction_section(self, tmp_path):
        run_info = self._make_run_info()
        path = write_results_md(str(tmp_path), run_info)
        content = open(path).read()
        assert "Reproduction" in content
        assert "make full-data" in content

    def test_empty_run_info(self, tmp_path):
        path = write_results_md(str(tmp_path), {})
        assert os.path.exists(path)
