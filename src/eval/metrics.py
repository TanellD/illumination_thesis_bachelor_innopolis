"""
src/eval/metrics.py
===================
Core metric suite shared across Stage 1 and Stage 2 evaluation.

All public functions use the canonical CSV key names defined in CLAUDE.md:
    auc, eer, fpr_at_tpr95, ece, brier

ECE uses 15 uniform bins, matching the original scripts.
Label convention throughout: 1 = real, 0 = fake; scores are p(real).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

ECE_BINS: int = 15


def _eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def _fpr_at_tpr(labels: np.ndarray, scores: np.ndarray,
                tpr_target: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(tpr >= tpr_target)[0]
    return float(fpr[idx[0]]) if len(idx) else 1.0


def _ece(labels: np.ndarray, scores: np.ndarray,
         n_bins: int = ECE_BINS) -> float:
    """Expected Calibration Error, equal-width bins.

    Bins are (lo, hi] with the first bin being [0, hi].  This matches the
    original main_tables_one_script.py convention (scores > lo and <= hi).
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores > lo) & (scores <= hi)
        if mask.sum():
            ece += mask.mean() * abs(scores[mask].mean() - (labels[mask] == 1).mean())
    return float(ece)


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Return the five-metric dict for a single (model, dataset) slice.

    Keys: auc, eer, fpr_at_tpr95, ece, brier.
    All metrics return nan when labels are single-class.
    """
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    nan5 = dict(auc=float("nan"), eer=float("nan"),
                fpr_at_tpr95=float("nan"), ece=float("nan"),
                brier=float("nan"), n=int(len(labels)))
    if len(np.unique(labels)) < 2:
        return nan5
    return dict(
        auc=float(roc_auc_score(labels, scores)),
        eer=_eer(labels, scores),
        fpr_at_tpr95=_fpr_at_tpr(labels, scores, 0.95),
        ece=_ece(labels, scores),
        brier=float(brier_score_loss(labels, scores)),
        n=int(len(labels)),
    )
