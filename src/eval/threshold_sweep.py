"""
src/eval/threshold_sweep.py
===========================
99-threshold sweep over [0.01, 0.99] reporting balanced accuracy, MCC,
and Youden's J at each threshold; plus the argmax of each.

F1 is deliberately NOT reported here for cross-dataset comparison — it is
degenerate under the class imbalance studied.  F1 is computed in the
original sweep_threshold.py for operating-point analysis only; that logic
lives in scripts/FailureTaxonomy/sweep_threshold.py.

Output of `sweep_thresholds()` is a DataFrame with one row per threshold
and columns: threshold, bal_acc, mcc, youden_j, tpr, tnr, fpr, fnr,
n_fail, fail_rate.

`find_optima()` reduces this to a dict with the argmax threshold and value
for each of bal_acc, mcc, youden_j.

Label convention: 1 = real, 0 = fake.  score > threshold → predict real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

THRESHOLDS = np.linspace(0.01, 0.99, 99)


def sweep_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray = THRESHOLDS,
) -> pd.DataFrame:
    """Per-threshold metric table.

    Parameters
    ----------
    scores     : (N,) p(real)
    labels     : (N,) int — 1 real / 0 fake
    thresholds : (T,) grid; defaults to 99 evenly-spaced values in [0.01, 0.99]

    Returns
    -------
    DataFrame with columns:
        threshold, bal_acc, mcc, youden_j, tpr, tnr, fpr, fnr,
        n, n_fail, fail_rate
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n = len(scores)
    n_real = int((labels == 1).sum())
    n_fake = int((labels == 0).sum())

    rows = []
    for t in thresholds:
        preds = (scores > t).astype(np.int32)
        fail = preds != labels
        n_fail = int(fail.sum())
        fail_rate = n_fail / n if n else float("nan")

        # FP = real predicted fake; FN = fake predicted real.
        fp = int(((preds == 0) & (labels == 1)).sum())
        fn = int(((preds == 1) & (labels == 0)).sum())
        fpr = fp / n_real if n_real else float("nan")   # miss rate on real
        fnr = fn / n_fake if n_fake else float("nan")   # miss rate on fake
        tpr = 1.0 - fpr   # real classified as real
        tnr = 1.0 - fnr   # fake classified as fake
        youden_j = float(tpr + tnr - 1.0)

        try:
            bal_acc = float(balanced_accuracy_score(labels, preds))
        except Exception:
            bal_acc = float("nan")

        try:
            mcc = float(matthews_corrcoef(labels, preds))
        except Exception:
            mcc = float("nan")

        rows.append({
            "threshold": float(t),
            "bal_acc": bal_acc,
            "mcc": mcc,
            "youden_j": youden_j,
            "tpr": float(tpr),
            "tnr": float(tnr),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "n": n,
            "n_fail": n_fail,
            "fail_rate": fail_rate,
        })
    return pd.DataFrame(rows)


def find_optima(curve: pd.DataFrame) -> dict:
    """Return the argmax threshold and value for bal_acc, mcc, youden_j.

    Also returns fail_rate at threshold-0.5 and at the argmax-bal_acc threshold.

    Keys returned:
        max_bal_acc, thresh_at_max_bal_acc,
        max_mcc,     thresh_at_max_mcc,
        max_youden_j, thresh_at_max_youden_j,
        fail_rate_at_0.5, fail_rate_at_opt_bal_acc
    """
    out: dict = {}
    for metric in ("bal_acc", "mcc", "youden_j"):
        valid = curve.dropna(subset=[metric])
        if len(valid) == 0:
            out[f"max_{metric}"] = float("nan")
            out[f"thresh_at_max_{metric}"] = float("nan")
            continue
        idx = valid[metric].idxmax()
        out[f"max_{metric}"] = float(valid.loc[idx, metric])
        out[f"thresh_at_max_{metric}"] = float(valid.loc[idx, "threshold"])

    closest_05 = curve.iloc[(curve["threshold"] - 0.5).abs().argmin()]
    out["fail_rate_at_0.5"] = float(closest_05["fail_rate"])

    opt_thresh = out.get("thresh_at_max_bal_acc", float("nan"))
    if not np.isnan(opt_thresh):
        row_opt = curve.iloc[(curve["threshold"] - opt_thresh).abs().argmin()]
        out["fail_rate_at_opt_bal_acc"] = float(row_opt["fail_rate"])
    else:
        out["fail_rate_at_opt_bal_acc"] = float("nan")

    return out
