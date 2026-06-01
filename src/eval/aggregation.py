"""
src/eval/aggregation.py
=======================
Video-level aggregation and cross-seed / cross-model statistical tests.

The grouping key for video-level aggregation is always
    (dataset, source, video_id)
NOT just video_id.  See KNOWN_QUIRKS.md #12.

SBI/T-SBI fakes share the source-video's video_id with the real frames they
were derived from; using a bare video_id would collapse real and fake rows
into the same group and inflate AUC.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.eval.metrics import compute_metrics


# ── video-level aggregation ────────────────────────────────────────────────────

def aggregate_to_video(
    scores: np.ndarray,
    labels: np.ndarray,
    sources: List[str],
    video_ids: List[str],
    dataset: Optional[str] = None,
    strategy: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Collapse per-frame scores to per-(source, video_id) scores.

    Parameters
    ----------
    scores    : (N,) float — p(real) per frame
    labels    : (N,) int  — 0 fake / 1 real per frame
    sources   : length-N source tags (e.g. "ffpp_real", "ffpp_fake_Deepfakes")
    video_ids : length-N video identifiers
    dataset   : optional dataset name; if given it is prepended to the key
                so that keys become (dataset, source, video_id) tuples.
                When aggregating a single-dataset slice, passing dataset=None
                is fine; keys will be (source, video_id) tuples.
    strategy  : 'mean' | 'max' | 'vote'

    Returns
    -------
    v_scores : (G,) aggregated score per group
    v_labels : (G,) majority label per group
    v_keys   : list of key tuples, in the same order as v_scores / v_labels
    """
    df = pd.DataFrame({"source": sources, "video_id": video_ids,
                       "score": scores, "label": labels})
    if dataset is not None:
        df["dataset"] = dataset
        group_cols = ["dataset", "source", "video_id"]
    else:
        group_cols = ["source", "video_id"]

    g = df.groupby(group_cols, sort=True)

    if strategy == "mean":
        agg_s = g["score"].mean()
    elif strategy == "max":
        agg_s = g["score"].max()
    elif strategy == "vote":
        agg_s = g["score"].apply(lambda s: float((s > 0.5).mean()))
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy!r}")

    agg_l = g["label"].agg(lambda s: int(s.mode().iloc[0]))
    keys = list(agg_s.index)
    return agg_s.to_numpy(), agg_l.to_numpy(), keys


def compute_video_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    sources: List[str],
    video_ids: List[str],
    dataset: Optional[str] = None,
    strategy: str = "mean",
) -> dict:
    """Aggregate to video level then compute the five-metric dict."""
    v_s, v_l, _ = aggregate_to_video(scores, labels, sources, video_ids,
                                      dataset=dataset, strategy=strategy)
    return compute_metrics(v_l, v_s)


# ── cross-seed aggregation ─────────────────────────────────────────────────────

def aggregate_seeds(results_list: List[dict]) -> Tuple[dict, dict]:
    """Arithmetic mean and unbiased std over per-seed metric dicts.

    Parameters
    ----------
    results_list : list of metric dicts (from compute_metrics / compute_video_metrics).
                   All dicts must have the same keys.

    Returns
    -------
    mean_dict : dict with the same keys, values = arithmetic mean across seeds
    std_dict  : dict with the same keys, values = sample std across seeds
                (ddof=1; nan when fewer than 2 seeds)
    """
    if not results_list:
        raise ValueError("results_list is empty")
    keys = [k for k in results_list[0] if k != "n"]
    mean_dict: dict = {}
    std_dict: dict = {}
    for k in keys:
        vals = np.array([r[k] for r in results_list], dtype=float)
        mean_dict[k] = float(np.nanmean(vals))
        std_dict[k] = float(np.nanstd(vals, ddof=1)) if len(vals) >= 2 else float("nan")
    # Propagate n from the first dict (they should all be equal)
    if "n" in results_list[0]:
        mean_dict["n"] = results_list[0]["n"]
    return mean_dict, std_dict


# ── Wilcoxon + Bonferroni ──────────────────────────────────────────────────────

def wilcoxon_bonferroni(
    a_scores: np.ndarray,
    a_labels: np.ndarray,
    b_scores: np.ndarray,
    b_labels: np.ndarray,
    n_comparisons: int = 1,
    alpha: float = 0.05,
) -> dict:
    """Paired Wilcoxon signed-rank test on per-sample squared errors.

    a_* and b_* must be aligned (same units, same order).  The test
    null hypothesis is that the distribution of squared-error differences
    is symmetric around zero.

    Bonferroni correction multiplies the raw p-value by n_comparisons and
    caps at 1.0.

    Returns keys: stat, p_value, p_corrected, effect_d (Cohen's d on the
    difference distribution), mean_brier_A, mean_brier_B, better ('A' or
    'B'), significant (bool), n.
    """
    a_scores = np.asarray(a_scores, dtype=np.float64)
    a_labels = np.asarray(a_labels, dtype=np.int32)
    b_scores = np.asarray(b_scores, dtype=np.float64)
    b_labels = np.asarray(b_labels, dtype=np.int32)

    e_a = (a_scores - a_labels) ** 2
    e_b = (b_scores - b_labels) ** 2
    diff = e_a - e_b

    try:
        stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p = float("nan"), 1.0

    p_corr = min(float(p) * n_comparisons, 1.0)
    sd = float(np.std(diff))
    d = float(np.mean(diff) / (sd + 1e-12))
    mean_a = float(np.mean(e_a))
    mean_b = float(np.mean(e_b))
    better = "A" if mean_a < mean_b else "B"

    return dict(
        stat=float(stat),
        p_value=float(p),
        p_corrected=p_corr,
        effect_d=d,
        mean_brier_A=mean_a,
        mean_brier_B=mean_b,
        better=better,
        significant=bool(p_corr < alpha),
        n=int(len(diff)),
    )


def align_paired_video(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    strategy: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Aggregate A and B to video level independently, then align on
    the intersection of (source, video_id) keys.

    Each dataframe must have columns: score, label, source, video_id.

    Returns (a_scores, a_labels, b_scores, b_labels, n_common).
    """
    def _agg(df):
        return aggregate_to_video(
            df["score"].to_numpy(),
            df["label"].to_numpy().astype(int),
            df["source"].astype(str).tolist(),
            df["video_id"].astype(str).tolist(),
            strategy=strategy,
        )

    a_s, a_l, a_k = _agg(df_a)
    b_s, b_l, b_k = _agg(df_b)

    a_map = {k: i for i, k in enumerate(a_k)}
    b_map = {k: i for i, k in enumerate(b_k)}
    common = sorted(set(a_map) & set(b_map))
    if not common:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, 0

    ia = np.array([a_map[k] for k in common])
    ib = np.array([b_map[k] for k in common])
    return a_s[ia], a_l[ia], b_s[ib], b_l[ib], len(common)
