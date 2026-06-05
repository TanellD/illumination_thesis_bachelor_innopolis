"""
src/report/figures.py
=====================
Figure generators for the thesis (G.2).

All functions produce deterministic PNG files at 150 dpi.  They are
CPU-only — no GPU or model weights required.

Figures produced:
    reliability_diagrams      Stage 2 reliability diagrams (3 panels × 5 curves)
    robustness_curves         AUC vs perturbation strength (multi-panel)
    stage1_auc_barplot        Stage 1 in-distribution + cross-dataset AUC bars

Ported from:
    scripts/visualizers/stage2_calibration.py
    scripts/visualizers/stage12_robustness_curves.py
    scripts/visualizers/stage1_barplot.py
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── model colour palette (shared across all figures) ─────────────────────────

MODEL_COLORS: Dict[str, str] = {
    "RGB-Only":            "#666666",
    "Residual-Only":       "#8B6F47",
    "Late-Fusion":         "#4A6FA5",
    "StatNoise-Fusion":    "#C9610F",
    "ResAware-Fusion":     "#7B2D8E",
    "A":                   "#185FA5",
    "B_pure":              "#993C1D",
    "B_mix":               "#0F6E56",
    "C_pure":              "#7B2D8E",
    "C_mix":               "#C9610F",
    "HIGHDL":              "#0F6E56",
    "LOWDL":               "#993C1D",
    "SBI-EfficientNet-B4": "#000000",
}

REGIME_STYLE: Dict[str, dict] = {
    "A":      dict(color="#185FA5", lw=2.5, ls="-",  marker="o", ms=5,
                   label="A  – baseline-mixed"),
    "B_pure": dict(color="#993C1D", lw=2.0, ls="--", marker="s", ms=4,
                   label="B_pure – SBI pure"),
    "B_mix":  dict(color="#0F6E56", lw=2.0, ls="--", marker="^", ms=4,
                   label="B_mix  – SBI mixed"),
    "C_pure": dict(color="#7B2D8E", lw=2.0, ls="-.", marker="D", ms=4,
                   label="C_pure – T-SBI pure"),
    "C_mix":  dict(color="#C9610F", lw=2.5, ls="-.", marker="P", ms=5,
                   label="C_mix  – T-SBI mixed"),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Wrote {path}")


# ── reliability diagrams ──────────────────────────────────────────────────────

def _ece_simple(labels: np.ndarray, scores: np.ndarray,
                n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(scores)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(scores[mask].mean() - labels[mask].mean())
    return float(ece)


def plot_reliability_diagrams(
    cache_root: str,
    datasets: List[str],
    regimes: Optional[List[str]] = None,
    out_path: str = "reliability_diagrams.png",
    n_bins: int = 15,
    show_gap: bool = True,
) -> None:
    """Plot Stage 2 frame-level reliability diagrams.

    Reads inference NPZ from <cache_root>/inference/<regime>_<dataset>.npz.

    Parameters
    ----------
    cache_root : root of DiskCache (contains inference/)
    datasets   : list of dataset names matching DiskCache keys
    regimes    : list of model names; defaults to the 5 Stage 2 regimes
    out_path   : output PNG path
    n_bins     : ECE / calibration bins (15 to match thesis)
    show_gap   : shade region between calibration curve and diagonal
    """
    from sklearn.calibration import calibration_curve

    regimes = regimes or list(REGIME_STYLE.keys())
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 5.5), sharey=True)
    fig.patch.set_facecolor("#f8f9fa")
    if n_ds == 1:
        axes = [axes]

    any_data = False
    for ax, ds_name in zip(axes, datasets):
        ax.set_facecolor("#f8f9fa")
        ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1.5,
                alpha=0.8, label="Perfect calibration", zorder=1)

        for mn in regimes:
            npz_path = Path(cache_root) / "inference" / f"{mn}_{ds_name}.npz"
            if not npz_path.exists():
                continue
            try:
                data = np.load(str(npz_path), allow_pickle=True)
                scores = np.asarray(data["scores"], dtype=np.float32)
                labels = np.asarray(data["labels"], dtype=np.int32)
            except Exception as exc:
                logger.warning(f"Failed reading {npz_path}: {exc}")
                continue
            if len(np.unique(labels)) < 2:
                continue

            try:
                frac_pos, mean_pred = calibration_curve(
                    labels, scores, n_bins=n_bins, strategy="uniform")
            except Exception:
                continue

            ece  = _ece_simple(labels, scores, n_bins)
            st   = REGIME_STYLE.get(mn, dict(color="grey", lw=1.5, ls="-",
                                              marker=".", ms=4, label=mn))
            ax.plot(mean_pred, frac_pos,
                    color=st["color"], lw=st["lw"], ls=st["ls"],
                    marker=st["marker"], markersize=st["ms"],
                    label=f"{st['label']}  (ECE={ece:.3f})", zorder=4)
            if show_gap:
                ax.fill_between(mean_pred, frac_pos, mean_pred,
                                alpha=0.07, color=st["color"], zorder=2)
            any_data = True

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability", fontsize=10)
        ax.set_title(ds_name, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7.5, loc="upper left", frameon=True,
                  framealpha=0.88, edgecolor="#cccccc", handlelength=2.5)

    axes[0].set_ylabel("Fraction of positives (real faces)", fontsize=10)
    fig.suptitle(
        "Stage 2 — Frame-level reliability diagrams\n"
        "Calibration per dataset and training regime",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _savefig(fig, out_path)
    if not any_data:
        logger.warning("No data found for reliability diagrams.")


# ── robustness curves ─────────────────────────────────────────────────────────

def plot_robustness_curves(
    robustness_csv: str,
    out_path: str = "robustness_curves.png",
    families: Optional[List[str]] = None,
    metric: str = "auc",
) -> None:
    """Line plots: metric vs perturbation strength, one panel per family.

    Reads the long-form robustness_grid.csv produced by src/robustness/grid.py.

    Parameters
    ----------
    robustness_csv : path to robustness_grid.csv
    out_path       : output PNG path
    families       : list of families to plot; defaults to all
    metric         : column from robustness_grid.csv (default 'auc')
    """
    if not os.path.exists(robustness_csv):
        logger.warning(f"robustness_csv not found: {robustness_csv}")
        return

    df = pd.read_csv(robustness_csv)
    if families is None:
        families = sorted(df["perturbation_family"].unique().tolist())

    n_fam = len(families)
    fig, axes = plt.subplots(1, n_fam, figsize=(4.5 * n_fam, 4.5), sharey=True)
    fig.patch.set_facecolor("white")
    if n_fam == 1:
        axes = [axes]

    FAMILY_XLABELS = {
        "jpeg":    "JPEG quality",
        "blur":    "Gaussian $\\sigma$",
        "dssharp": "Filter",
        "gamma":   "$\\gamma$",
        "resize":  "Resize factor",
    }

    for ax, fam in zip(axes, families):
        sub = df[df["perturbation_family"] == fam]
        models = sorted(sub["model"].unique())
        datasets = sorted(sub["dataset"].unique())

        for model in models:
            for ds in datasets:
                msub = sub[(sub["model"] == model) & (sub["dataset"] == ds)]
                if msub.empty:
                    continue
                # Sort x-axis by numeric value where possible
                try:
                    msub = msub.copy()
                    msub["_x"] = pd.to_numeric(msub["perturbation_value"],
                                                errors="coerce")
                    msub = msub.sort_values("_x")
                    x    = msub["_x"].tolist()
                except Exception:
                    x = msub["perturbation_value"].tolist()
                y = msub[metric].tolist()
                color = MODEL_COLORS.get(model, "grey")
                label = f"{model}/{ds}" if len(datasets) > 1 else model
                ax.plot(x, y, marker="o", ms=4, lw=1.8,
                        color=color, label=label)

        ax.set_xlabel(FAMILY_XLABELS.get(fam, fam), fontsize=10)
        ax.set_title(fam, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, loc="best")
        if metric == "auc":
            ax.set_ylim(0.4, 1.0)

    axes[0].set_ylabel(metric.upper(), fontsize=10)
    fig.suptitle(f"Robustness grid — {metric.upper()} vs perturbation",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_path)


# ── Stage 1 AUC bar plot ──────────────────────────────────────────────────────

def plot_stage1_auc_bars(
    metrics_csv: str,
    out_path: str = "stage1_auc_bars.png",
    datasets_in_dist:    Optional[List[str]] = None,
    datasets_cross_dist: Optional[List[str]] = None,
) -> None:
    """Grouped bar chart: Stage 1 model AUC, in-distribution vs cross-dataset.

    Reads a CSV with columns: model, dataset, auc (can be from build_T3 output).

    Parameters
    ----------
    metrics_csv         : path to CSV with model, dataset, auc columns
    out_path            : output PNG path
    datasets_in_dist    : dataset names for left-panel group
    datasets_cross_dist : dataset names for right-panel group
    """
    if not os.path.exists(metrics_csv):
        logger.warning(f"metrics_csv not found: {metrics_csv}")
        return

    df = pd.read_csv(metrics_csv)

    # T3 CSV is wide format: model, FF++, Celeb-DF, DFDC, DFF
    # Melt to long format: model, dataset, auc
    if "model" in df.columns and "auc" not in df.columns:
        dataset_cols = [c for c in df.columns if c != "model"]
        df = df.melt(id_vars="model", value_vars=dataset_cols,
                     var_name="dataset", value_name="auc")

    in_dist    = datasets_in_dist    or ["FF++"]
    cross_dist = datasets_cross_dist or ["Celeb-DF", "DFDC", "DFF"]

    models = sorted(df["model"].unique())
    x      = np.arange(len(models))
    width  = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, group, title in [
        (axes[0], in_dist,    "In-distribution (FF++)"),
        (axes[1], cross_dist, "Cross-dataset"),
    ]:
        for j, ds in enumerate(group):
            sub = df[df["dataset"] == ds]
            aucs = [float(sub.loc[sub["model"] == m, "auc"].values[0])
                    if not sub.loc[sub["model"] == m, "auc"].empty
                    else np.nan
                    for m in models]
            offset = (j - len(group) / 2.0 + 0.5) * width
            colors = [MODEL_COLORS.get(m, "grey") for m in models]
            bars = ax.bar(x + offset, aucs, width, label=ds, color=colors,
                          alpha=0.85, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("AUC", fontsize=10)
        ax.set_ylim(0.4, 1.0)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Stage 1 — AUC by model and dataset",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_path)
