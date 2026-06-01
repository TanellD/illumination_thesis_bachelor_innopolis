"""
src/taxonomy/contingency.py
============================
Statistical tests for the failure taxonomy (experiments F.2–F.4).

F.2 — Quantile binning: bin continuous attributes within each dataset.
F.3 — Contingency tests: chi-square + Cramér's V per (model, dataset, axis).
F.4 — Cross-model failure agreement: Jaccard similarity + pairwise lift.

Ported verbatim from:
  scripts/FailureTaxonomy/models_on_taxonomy..py  (chi2, McNemar)
  scripts/FailureTaxonomy/aggregate_unique_failures.py  (Jaccard, agreement)
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

FAILURE_THRESHOLD = 0.5

# Cramér's V effect size labels
def _effect_label(v: float) -> str:
    if v < 0.1:  return "negligible"
    if v < 0.3:  return "small"
    if v < 0.5:  return "medium"
    return "large"


# ── F.2 — quantile binning ────────────────────────────────────────────────────

def quantile_bin_within_dataset(
    df: pd.DataFrame,
    attribute_col: str,
    n_quantiles: int = 4,
    new_col: Optional[str] = None,
) -> pd.DataFrame:
    """Bin a continuous attribute into n_quantiles within each dataset.

    This controls for dataset-specific distributional shifts so that
    "Q1" means "lowest quartile in THIS dataset", not globally.

    Parameters
    ----------
    df            : DataFrame with columns [dataset, <attribute_col>, ...]
    attribute_col : name of the continuous column to bin
    n_quantiles   : number of quantile bins (default 4 = quartiles)
    new_col       : output column name; defaults to attribute_col + "_q"

    Returns
    -------
    df with an additional column for the quantile bin label ("Q1".."Q4")
    """
    if new_col is None:
        new_col = attribute_col + "_q"
    df = df.copy()
    df[new_col] = "unknown"
    for dataset, sub in df.groupby("dataset"):
        valid = sub[attribute_col].notna()
        if valid.sum() < n_quantiles:
            continue
        try:
            labels = [f"Q{i+1}" for i in range(n_quantiles)]
            binned = pd.qcut(sub.loc[valid, attribute_col],
                              q=n_quantiles, labels=labels, duplicates="drop")
            df.loc[sub.index[valid], new_col] = binned.astype(str)
        except Exception:
            pass
    return df


# ── F.3 — chi-square + Cramér's V ────────────────────────────────────────────

def chi_square_cramers(
    long_df: pd.DataFrame,
    attribute_cols: List[str],
    score_col:    str = "score",
    label_col:    str = "label",
    threshold:    float = FAILURE_THRESHOLD,
) -> pd.DataFrame:
    """For each (model, dataset, attribute_col): chi-square test of
    independence between failure status and attribute bin.

    Parameters
    ----------
    long_df        : DataFrame with columns:
                     model, dataset, face_crop_path, score, label,
                     + any attribute bin columns
    attribute_cols : list of bin columns to test (e.g. ["blur_bin", "pose_bin"])
    score_col      : column with p(real) scores
    label_col      : column with 0/1 labels
    threshold      : decision threshold for failure

    Returns
    -------
    DataFrame with columns:
        model, dataset, axis, chi2, dof, p_value, cramers_v, n, effect
    """
    df = long_df.copy()
    df["is_failure"] = (
        (df[score_col] > threshold).astype(int) != df[label_col].astype(int)
    ).astype(int)

    rows = []
    for (model, dataset), sub_md in df.groupby(["model", "dataset"]):
        for col in attribute_cols:
            if col not in sub_md.columns:
                continue
            sub = sub_md[[col, "is_failure"]].dropna(subset=[col])
            sub = sub[~sub[col].isin(["unknown", "nan", "NaN", "Unknown"])]
            if len(sub) < 20:
                continue
            ct = pd.crosstab(sub[col], sub["is_failure"])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            chi2_stat, p, dof, _ = chi2_contingency(ct.values)
            n          = int(ct.values.sum())
            cramers_v  = float((chi2_stat / (n * (min(ct.shape) - 1))) ** 0.5)
            rows.append({
                "model":     model,
                "dataset":   dataset,
                "axis":      col,
                "chi2":      float(chi2_stat),
                "dof":       int(dof),
                "p_value":   float(p),
                "cramers_v": cramers_v,
                "n":         n,
                "effect":    _effect_label(cramers_v),
            })
    return pd.DataFrame(rows)


def per_bin_failure_rates(
    long_df: pd.DataFrame,
    attribute_cols: List[str],
    score_col:    str = "score",
    label_col:    str = "label",
    threshold:    float = FAILURE_THRESHOLD,
) -> pd.DataFrame:
    """Per-bin failure rate for each (model, dataset, axis, bin).

    Returns a tidy DataFrame with columns:
        model, dataset, axis, bin, n, n_fail, fail_rate
    """
    df = long_df.copy()
    df["is_failure"] = (
        (df[score_col] > threshold).astype(int) != df[label_col].astype(int)
    ).astype(int)

    rows = []
    for (model, dataset), sub_md in df.groupby(["model", "dataset"]):
        for col in attribute_cols:
            if col not in sub_md.columns:
                continue
            for bin_val, sub_b in sub_md.groupby(col):
                if str(bin_val) in ("unknown", "nan", "NaN", "Unknown"):
                    continue
                n      = len(sub_b)
                n_fail = int(sub_b["is_failure"].sum())
                rows.append({
                    "model":     model,
                    "dataset":   dataset,
                    "axis":      col,
                    "bin":       str(bin_val),
                    "n":         n,
                    "n_fail":    n_fail,
                    "fail_rate": round(n_fail / n, 4) if n else float("nan"),
                })
    return pd.DataFrame(rows)


# ── F.4 — cross-model failure agreement ──────────────────────────────────────

def failure_jaccard(
    long_df: pd.DataFrame,
    model_col:  str = "model",
    score_col:  str = "score",
    label_col:  str = "label",
    path_col:   str = "face_crop_path",
    threshold:  float = FAILURE_THRESHOLD,
) -> pd.DataFrame:
    """Per (dataset, model_A, model_B): Jaccard index and lift of failure sets.

    Jaccard = |A ∩ B| / |A ∪ B|  (1 = identical failures, 0 = disjoint)
    Lift    = P(A∩B) / [P(A)·P(B)]  (>1 = correlated blind spots)

    Returns DataFrame with columns:
        dataset, model_A, model_B, n_total,
        n_A_fail, n_B_fail, n_both_fail, n_either_fail,
        jaccard, lift
    """
    df = long_df.copy()
    df["is_failure"] = (
        (df[score_col] > threshold).astype(int) != df[label_col].astype(int)
    ).astype(int)

    rows = []
    for dataset, sub_d in df.groupby("dataset"):
        pivot = sub_d.pivot_table(
            index=path_col, columns=model_col,
            values="is_failure", aggfunc="first",
        )
        n_total = len(pivot)
        models  = sorted(pivot.columns.tolist())
        fails   = {m: pivot[m].fillna(0).astype(bool).values for m in models}

        for m_a, m_b in combinations(models, 2):
            a  = fails[m_a]
            b  = fails[m_b]
            na = int(a.sum()); nb = int(b.sum())
            intersection = int((a & b).sum())
            union        = int((a | b).sum())
            jaccard      = intersection / union if union else float("nan")
            p_a  = na / n_total if n_total else 0.0
            p_b  = nb / n_total if n_total else 0.0
            p_ab = intersection / n_total if n_total else 0.0
            lift = (p_ab / (p_a * p_b)) if (p_a > 0 and p_b > 0) else float("nan")
            rows.append({
                "dataset":       dataset,
                "model_A":       m_a,
                "model_B":       m_b,
                "n_total":       n_total,
                "n_A_fail":      na,
                "n_B_fail":      nb,
                "n_both_fail":   intersection,
                "n_either_fail": union,
                "jaccard":       round(jaccard, 4)
                                 if not np.isnan(jaccard) else None,
                "lift":          round(lift, 3)
                                 if not np.isnan(lift)    else None,
            })
    return pd.DataFrame(rows)


def agreement_distribution(
    long_df: pd.DataFrame,
    model_col:  str = "model",
    score_col:  str = "score",
    label_col:  str = "label",
    path_col:   str = "face_crop_path",
    threshold:  float = FAILURE_THRESHOLD,
) -> pd.DataFrame:
    """Per dataset: how many samples are failed by 0, 1, 2, ... models.

    Returns DataFrame with columns:
        dataset, n_models_failed, n_samples, pct_of_dataset
    """
    df = long_df.copy()
    df["is_failure"] = (
        (df[score_col] > threshold).astype(int) != df[label_col].astype(int)
    ).astype(int)

    rows = []
    for dataset, sub_d in df.groupby("dataset"):
        pivot = sub_d.pivot_table(
            index=path_col, columns=model_col,
            values="is_failure", aggfunc="first",
        ).fillna(0)
        n_failed = pivot.sum(axis=1).astype(int)
        total    = len(pivot)
        for k, count in n_failed.value_counts().sort_index().items():
            rows.append({
                "dataset":         dataset,
                "n_models_failed": int(k),
                "n_samples":       int(count),
                "pct_of_dataset":  round(100.0 * count / total, 3),
            })
    return pd.DataFrame(rows)


def unique_failures_table(
    long_df: pd.DataFrame,
    model_col:  str = "model",
    score_col:  str = "score",
    label_col:  str = "label",
    path_col:   str = "face_crop_path",
    threshold:  float = FAILURE_THRESHOLD,
) -> pd.DataFrame:
    """One row per (path, dataset): which models failed, total count.

    Useful for deep-diving model-specific failure cases.
    """
    df = long_df.copy()
    df["is_failure"] = (
        (df[score_col] > threshold).astype(int) != df[label_col].astype(int)
    ).astype(int)

    pivot = df.pivot_table(
        index=["dataset", path_col, label_col],
        columns=model_col,
        values="is_failure",
        aggfunc="first",
    ).reset_index()
    model_cols = [c for c in pivot.columns
                  if c not in ("dataset", path_col, label_col)]
    pivot["n_models_failed"] = pivot[model_cols].fillna(0).sum(axis=1).astype(int)
    pivot["all_failed"]      = (pivot["n_models_failed"] == len(model_cols)).astype(int)
    pivot["only_one_failed"] = (pivot["n_models_failed"] == 1).astype(int)
    return pivot
