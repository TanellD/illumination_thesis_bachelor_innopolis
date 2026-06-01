"""
src/eval/bootstrap.py
=====================
95% bootstrap confidence intervals for AUC (and any metric function).

Resampling unit:
  - Cross-dataset sets (ff++, celebdf, dfdc): resample over *videos*
    using the (source, video_id) grouping key.  This mirrors KNOWN_QUIRKS
    #12: we must never mix real and SBI-derived fakes in the same group.
  - DFF (image-only dataset): resample over *samples* directly, since
    there is no meaningful video grouping.

Both modes produce 1 000 resamples by default (n_resamples=1000) and
return 2.5th / 97.5th percentile CIs (level=0.95).

Usage
-----
>>> from src.eval.bootstrap import bootstrap_auc_ci
>>> lo, hi = bootstrap_auc_ci(scores, labels, video_ids=vids, sources=srcs)
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from src.eval.metrics import compute_metrics, _ece, ECE_BINS
from sklearn.metrics import brier_score_loss, roc_auc_score


def _default_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def bootstrap_auc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    video_ids: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """95% CI on AUC via bootstrap.

    If video_ids and sources are provided, resamples over
    (source, video_id) groups (cross-dataset mode).
    Otherwise resamples over individual samples (DFF mode).

    Returns (ci_lo, ci_hi).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    rng = _default_rng(seed)
    alpha = 1.0 - level

    if video_ids is not None and sources is not None:
        aucs = _bootstrap_over_videos(
            scores, labels,
            metric_fn=_auc_safe, n_resamples=n_resamples, rng=rng,
            sources=list(sources), video_ids=list(video_ids))
    else:
        aucs = _bootstrap_over_samples(
            scores, labels, metric_fn=_auc_safe,
            n_resamples=n_resamples, rng=rng)

    clean = aucs[~np.isnan(aucs)]
    if len(clean) == 0:
        return float("nan"), float("nan")
    lo = float(np.percentile(clean, 100.0 * alpha / 2))
    hi = float(np.percentile(clean, 100.0 * (1.0 - alpha / 2)))
    return lo, hi


def bootstrap_metrics_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    video_ids: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> dict:
    """95% CIs for AUC, Brier, and ECE simultaneously.

    Returns a dict with keys:
        auc, auc_ci_lo, auc_ci_hi,
        brier, brier_ci_lo, brier_ci_hi,
        ece, ece_ci_lo, ece_ci_hi,
        n_samples
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    rng = _default_rng(seed)
    alpha = 1.0 - level

    use_videos = video_ids is not None and sources is not None
    _bootstrap = _bootstrap_over_videos if use_videos else _bootstrap_over_samples

    def _auc(s, l):    return _auc_safe(s, l)
    def _brier(s, l):  return float(brier_score_loss(l, s))
    def _ece_fn(s, l): return float(_ece(l, s))

    kwargs: dict = {"n_resamples": n_resamples, "rng": rng}
    if use_videos:
        kwargs["sources"] = list(sources)
        kwargs["video_ids"] = list(video_ids)

    aucs   = _bootstrap(scores, labels, metric_fn=_auc,   **kwargs)
    briers = _bootstrap(scores, labels, metric_fn=_brier, **kwargs)
    eces   = _bootstrap(scores, labels, metric_fn=_ece_fn, **kwargs)

    def _ci(arr):
        c = arr[~np.isnan(arr)]
        if len(c) == 0:
            return float("nan"), float("nan")
        return (float(np.percentile(c, 100.0 * alpha / 2)),
                float(np.percentile(c, 100.0 * (1.0 - alpha / 2))))

    auc_lo, auc_hi   = _ci(aucs)
    br_lo,  br_hi    = _ci(briers)
    ece_lo, ece_hi   = _ci(eces)

    return dict(
        n_samples=int(len(scores)),
        auc=_auc_safe(scores, labels),
        auc_ci_lo=auc_lo, auc_ci_hi=auc_hi,
        brier=float(brier_score_loss(labels, scores)),
        brier_ci_lo=br_lo, brier_ci_hi=br_hi,
        ece=float(_ece(labels, scores)),
        ece_ci_lo=ece_lo, ece_ci_hi=ece_hi,
    )


# ── internal helpers ──────────────────────────────────────────────────────────

def _auc_safe(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def _bootstrap_over_samples(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_fn: Callable,
    n_resamples: int,
    rng: np.random.Generator,
    **_,
) -> np.ndarray:
    """Resample over individual rows (DFF mode)."""
    n = len(scores)
    vals = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        vals[b] = metric_fn(scores[idx], labels[idx])
    return vals


def _bootstrap_over_videos(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_fn: Callable,
    n_resamples: int,
    rng: np.random.Generator,
    sources: List[str],
    video_ids: List[str],
    **_,
) -> np.ndarray:
    """Resample over (source, video_id) groups (cross-dataset mode)."""
    import pandas as pd
    df = pd.DataFrame({"score": scores, "label": labels,
                       "source": sources, "video_id": video_ids})
    groups = df.groupby(["source", "video_id"])
    group_keys = list(groups.groups.keys())
    n_groups = len(group_keys)

    vals = np.empty(n_resamples)
    for b in range(n_resamples):
        selected = rng.integers(0, n_groups, size=n_groups)
        rows = pd.concat([groups.get_group(group_keys[i]) for i in selected],
                         ignore_index=True)
        vals[b] = metric_fn(rows["score"].to_numpy(),
                            rows["label"].to_numpy().astype(np.int32))
    return vals
