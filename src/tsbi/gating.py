"""
src/tsbi/gating.py
==================
Pair-level illumination gating and dL-quartile manifest splitting.

Gating logic (CLAUDE.md §C.2):
  Accept a pair if:
    dL_mean >= 6.0  OR  dL_std >= 3.0
  If no pair clears the strict gate for a source video, accept the pair with
  the largest available delta (relax fallback) and flag it with illum_relaxed=1.

dL-quartile split (CLAUDE.md §C.4):
  Build HIGHDL and LOWDL manifest variants by splitting T-SBI rows on the
  q75/q25 dL_mean thresholds computed from the TRAIN split only, then applied
  to both train and val.

Ported from scripts/T_SBI/build_low_high_dl_manifests.py.
"""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

# ── gating constants — do NOT change without updating KNOWN_QUIRKS.md ────────
MIN_ILLUM_DELTA_MEAN: float = 6.0
MIN_ILLUM_DELTA_STD:  float = 3.0


def gate_pair(
    dL_mean: float,
    dL_std: float,
    min_dL_mean: float = MIN_ILLUM_DELTA_MEAN,
    min_dL_std: float  = MIN_ILLUM_DELTA_STD,
) -> bool:
    """Return True when a (src, tgt) pair passes the strict illumination gate."""
    return dL_mean >= min_dL_mean or dL_std >= min_dL_std


# ── dL-quartile manifest splitting ───────────────────────────────────────────

def load_dl_lookup(tsbi_csv: str) -> dict:
    """Build {face_crop_path: dL_mean} from a T-SBI labels CSV."""
    df = pd.read_csv(tsbi_csv)
    if "face_crop_path" not in df.columns or "illum_delta_L" not in df.columns:
        raise ValueError(
            "tsbi_labels.csv must have face_crop_path and illum_delta_L columns"
        )
    return dict(zip(df["face_crop_path"], df["illum_delta_L"]))


def split_by_dl_quartile(
    manifest_df: pd.DataFrame,
    dl_lookup: dict,
    high_threshold: Optional[float] = None,
    low_threshold:  Optional[float] = None,
    compute_thresholds: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Split T-SBI rows in a manifest into HIGH-dL and LOW-dL subsets.

    Non-T-SBI rows (real frames, SBI fakes) are kept identical in both subsets.

    Parameters
    ----------
    manifest_df       : the full manifest DataFrame
    dl_lookup         : dict mapping face_crop_path → dL_mean
    high_threshold    : dL_mean >= this → HIGH subset; required if not computing
    low_threshold     : dL_mean <= this → LOW subset;  required if not computing
    compute_thresholds: if True, compute q75/q25 from the T-SBI rows in this df

    Returns
    -------
    (high_df, low_df, used_high_threshold, used_low_threshold)
    """
    m = manifest_df.copy()

    # Identify T-SBI rows
    if "source" in m.columns:
        is_tsbi = (m["source"].astype(str) == "tsbi")
    else:
        is_tsbi = m["face_crop_path"].isin(dl_lookup)

    m["_dL"] = m["face_crop_path"].map(dl_lookup)
    tsbi_rows = m[is_tsbi].copy()

    n_missing = int(tsbi_rows["_dL"].isna().sum())
    if n_missing:
        import warnings
        warnings.warn(
            f"{n_missing}/{len(tsbi_rows)} T-SBI rows have no dL match "
            "— they will be excluded from both subsets"
        )
    tsbi_rows = tsbi_rows.dropna(subset=["_dL"])

    if compute_thresholds:
        if len(tsbi_rows) == 0:
            raise ValueError("No T-SBI rows with dL match — cannot compute thresholds")
        high_threshold = float(tsbi_rows["_dL"].quantile(0.75))
        low_threshold  = float(tsbi_rows["_dL"].quantile(0.25))

    if high_threshold is None or low_threshold is None:
        raise ValueError("Either provide thresholds or set compute_thresholds=True")

    tsbi_high = tsbi_rows[tsbi_rows["_dL"] >= high_threshold].drop(columns=["_dL"])
    tsbi_low  = tsbi_rows[tsbi_rows["_dL"] <= low_threshold].drop(columns=["_dL"])
    other     = m[~is_tsbi].drop(columns=["_dL"], errors="ignore")

    high_df = pd.concat([other, tsbi_high], ignore_index=True)
    low_df  = pd.concat([other, tsbi_low],  ignore_index=True)

    return high_df, low_df, float(high_threshold), float(low_threshold)


def build_highdl_lowdl_manifests(
    tsbi_csv: str,
    train_manifest_csv: str,
    val_manifest_csv: str,
    out_dir: str,
) -> dict:
    """Build the four HIGHDL/LOWDL manifest CSVs for experiment C.4.

    Thresholds (q75/q25) are computed from the TRAIN manifest only, then
    applied to both train and val — see CLAUDE.md §C.4.

    Returns dict of {filename: DataFrame}.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    dl = load_dl_lookup(tsbi_csv)
    train_df = pd.read_csv(train_manifest_csv)
    val_df   = pd.read_csv(val_manifest_csv)

    train_high, train_low, hi_thr, lo_thr = split_by_dl_quartile(
        train_df, dl, compute_thresholds=True
    )
    val_high, val_low, _, _ = split_by_dl_quartile(
        val_df, dl,
        high_threshold=hi_thr, low_threshold=lo_thr,
        compute_thresholds=False,
    )

    results = {
        "C_pure_HIGHDL_train.csv": train_high,
        "C_pure_HIGHDL_val.csv":   val_high,
        "C_pure_LOWDL_train.csv":  train_low,
        "C_pure_LOWDL_val.csv":    val_low,
    }
    for name, df in results.items():
        df.to_csv(os.path.join(out_dir, name), index=False)

    return results
