"""
src/report/tables.py
====================
Table builders for all ten thesis tables (T1–T10).

Each builder is a pure function:
    build_T<N>(preds_dir, cfg) -> (tidy_df, latex_str)

where preds_dir contains per-frame prediction CSVs written by inference
and cfg is a dict loaded from configs/eval/*.yaml.

Input CSV format (per frame):
    score, label, source, video_id, method, path

All builders are CPU-only — no model or GPU is needed.

Tables produced:
    T1  Stage 1 video-level on FF++ (§3.7.2)
    T2  Stage 1 fixed-fusion variants, 5-seed mean ± std (§3.7.2)
    T3  Stage 1 cross-dataset AUC (§3.7.5)
    T4  Stage 1 robustness: JPEG + blur (§3.7.6)
    T5  Stage 1 Wilcoxon signed-rank tests, Bonferroni-corrected (§3.7.8)
    T6  Five-regime cross-dataset AUC (§3.7.4 Stage 2)
    T7  Stage 2 calibration: ECE + Brier (§3.7.4)
    T8  dL-quartile FF++ per-method AUC (§3.7.4)
    T9  dL-quartile Celeb-DF aggregation strategies (§3.7.4)
    T10 T-SBI pair-sampling dL distribution statistics (§3.7.4)

Ported from scripts/FailureTaxonomy/main_tables_one_script.py.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.eval.metrics import compute_metrics
from src.eval.aggregation import aggregate_to_video, wilcoxon_bonferroni


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt(x: float, decimals: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{decimals}f}"


def _video_metrics(df: pd.DataFrame, strategy: str = "mean") -> dict:
    n = len(df)
    sources  = df["source"].astype(str).tolist()  if "source"   in df.columns else ["s"] * n
    vids     = df["video_id"].astype(str).tolist() if "video_id" in df.columns else [str(i) for i in range(n)]
    v_s, v_l, _ = aggregate_to_video(
        df["score"].to_numpy(),
        df["label"].to_numpy().astype(int),
        sources, vids,
        strategy=strategy,
    )
    return compute_metrics(v_l, v_s)


def _frame_metrics(df: pd.DataFrame) -> dict:
    return compute_metrics(
        df["label"].to_numpy().astype(int),
        df["score"].to_numpy(),
    )


def _per_method_metrics(df: pd.DataFrame, level: str = "video") -> pd.DataFrame:
    """Per-FF++ manipulation method: (all reals) ∪ (fakes of that method)."""
    rows = []
    real_mask = df["label"].astype(int) == 1
    # method column may be absent if pred CSV was written without it
    if "method" not in df.columns:
        df = df.copy()
        df["method"] = "unknown"
    methods = sorted(
        set(df.loc[~real_mask, "method"].astype(str)) - {"unknown", ""}
    )
    for method in methods:
        mmask = (df["method"].astype(str) == method) & (~real_mask)
        sub = pd.concat([df[real_mask], df[mmask]], ignore_index=True)
        m = _video_metrics(sub) if level == "video" else _frame_metrics(sub)
        rows.append({"method": method, "level": level, **m})
    m_all = _video_metrics(df) if level == "video" else _frame_metrics(df)
    rows.append({"method": "OVERALL", "level": level, **m_all})
    return pd.DataFrame(rows)


def _read_pred(preds_dir: str, regime: str, mkey: str,
               ptag: str = "") -> Optional[pd.DataFrame]:
    fname = f"{regime}__{mkey}"
    if ptag:
        fname += f"__{ptag}"
    p = os.path.join(preds_dir, fname + ".csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def _latex_table(
    headers: List[str],
    rows: List[List[str]],
    caption: str = "",
    label: str = "",
    colspec: Optional[str] = None,
) -> str:
    if colspec is None:
        colspec = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\small",
        f"  \\begin{{tabular}}{{{colspec}}}",
        "    \\toprule",
        "    \\rowcolor{tblheader}",
        "    " + " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
        "    \\midrule",
    ]
    for r in rows:
        lines.append("    " + " & ".join(r) + " \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    if caption:
        lines.append(f"  \\caption{{{caption}}}")
    if label:
        lines.append(f"  \\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ── T1 ─────────────────────────────────────────────────────────────────────────

def build_T1(preds_dir: str, cfg: dict) -> Tuple[pd.DataFrame, str]:
    """T1: Stage 1 video-level on FF++ test split."""
    models = cfg["table_assignments"]["T1"]
    tidy, tex = [], []
    for regime in models:
        df = _read_pred(preds_dir, regime, "FF++_stage1")
        if df is None:
            tidy.append({"model": regime, "auc": np.nan, "eer": np.nan,
                          "fpr_at_tpr95": np.nan, "brier": np.nan, "ece": np.nan})
            tex.append([regime, "--", "--", "--", "--", "--"])
            continue
        m = _video_metrics(df, "mean")
        tidy.append({"model": regime, "auc": m["auc"],
                     "eer": m["eer"], "fpr_at_tpr95": m["fpr_at_tpr95"],
                     "brier": m["brier"], "ece": m["ece"]})
        tex.append([regime, _fmt(m["auc"]), _fmt(m["eer"]),
                    _fmt(m["fpr_at_tpr95"]), _fmt(m["brier"]), _fmt(m["ece"])])
    latex = _latex_table(
        headers=["Model", "AUC", "EER", "FPR@95", "Brier", "ECE"],
        rows=tex,
        caption="Stage~1 video-level results on FF++ in-distribution test partition.",
        label="tab:t1_stage1_ffpp",
    )
    return pd.DataFrame(tidy), latex


# ── T2 ─────────────────────────────────────────────────────────────────────────

def build_T2(preds_dir: str, cfg: dict) -> Tuple[pd.DataFrame, str]:
    """T2: Stage 1 fixed-fusion variants with 5-seed mean ± std."""
    spec = cfg["table_assignments"]["T2"]

    def _auc(regime: str, mkey: str) -> float:
        df = _read_pred(preds_dir, regime, mkey)
        return float("nan") if df is None else _video_metrics(df)["auc"]

    rgb_ff  = _auc(spec["baselines"][0], "FF++_stage1")
    rgb_cdf = _auc(spec["baselines"][0], "CelebDF")
    tidy, tex = [], []

    for regime in spec["baselines"]:
        ff  = _auc(regime, "FF++_stage1")
        cdf = _auc(regime, "CelebDF")
        ref = (regime == spec["baselines"][0])
        d_ff  = 0.0 if ref else (ff  - rgb_ff  if np.isfinite(ff)  else np.nan)
        d_cdf = 0.0 if ref else (cdf - rgb_cdf if np.isfinite(cdf) else np.nan)
        tidy.append({"model": regime, "ff_auc": ff, "ff_delta": d_ff,
                     "cdf_auc": cdf, "cdf_delta": d_cdf,
                     "seeds": 1, "std_ff": 0.0, "std_cdf": 0.0})
        tex.append([regime, _fmt(ff), "ref" if ref else _fmt(d_ff, 3),
                    _fmt(cdf), "ref" if ref else _fmt(d_cdf, 3)])

    for name, seed_regimes in spec.get("fixed", {}).items():
        ff_v  = [a for a in [_auc(r, "FF++_stage1") for r in seed_regimes] if np.isfinite(a)]
        cdf_v = [a for a in [_auc(r, "CelebDF")     for r in seed_regimes] if np.isfinite(a)]
        ff_m  = float(np.mean(ff_v))  if ff_v  else np.nan
        ff_s  = float(np.std(ff_v))   if ff_v  else np.nan
        cdf_m = float(np.mean(cdf_v)) if cdf_v else np.nan
        cdf_s = float(np.std(cdf_v))  if cdf_v else np.nan
        d_ff  = ff_m  - rgb_ff  if np.isfinite(ff_m)  else np.nan
        d_cdf = cdf_m - rgb_cdf if np.isfinite(cdf_m) else np.nan
        tidy.append({"model": name, "ff_auc": ff_m, "ff_delta": d_ff,
                     "cdf_auc": cdf_m, "cdf_delta": d_cdf,
                     "seeds": len(seed_regimes), "std_ff": ff_s, "std_cdf": cdf_s})
        tex.append([
            name,
            f"${_fmt(ff_m)} \\pm {_fmt(ff_s, 3)}$",
            _fmt(d_ff, 3),
            f"${_fmt(cdf_m)} \\pm {_fmt(cdf_s, 3)}$",
            _fmt(d_cdf, 3),
        ])

    latex = _latex_table(
        headers=["Model", "FF++ AUC", "$\\Delta$FF", "Celeb-DF AUC", "$\\Delta$CDF"],
        rows=tex,
        caption="Stage~1 fixed-fusion variants (5-seed mean~$\\pm$~std).",
        label="tab:t2_stage1_fusion",
    )
    return pd.DataFrame(tidy), latex


# ── T3 ─────────────────────────────────────────────────────────────────────────

def build_T3(preds_dir: str, cfg: dict) -> Tuple[pd.DataFrame, str]:
    """T3: Stage 1 cross-dataset AUC."""
    models = cfg["table_assignments"]["T3"]
    datasets = [("FF++", "FF++_stage1", "video"),
                ("Celeb-DF", "CelebDF", "video"),
                ("DFDC", "DFDC", "video"),
                ("DFF", "DFF", "frame")]
    tidy, tex = [], []
    for regime in models:
        row: dict = {"model": regime}
        tex_row = [regime]
        for ds_label, mkey, level in datasets:
            df = _read_pred(preds_dir, regime, mkey)
            if df is None:
                row[ds_label] = np.nan; tex_row.append("--")
            else:
                m = _frame_metrics(df) if level == "frame" else _video_metrics(df)
                row[ds_label] = m["auc"]; tex_row.append(_fmt(m["auc"]))
        tidy.append(row); tex.append(tex_row)
    latex = _latex_table(
        headers=["Model"] + [d[0] for d in datasets],
        rows=tex,
        colspec="l" + "r" * len(datasets),
        caption="Stage~1 cross-dataset AUC (video-level except DFF which is frame-level).",
        label="tab:t3_stage1_cross",
    )
    return pd.DataFrame(tidy), latex


# ── T4 ─────────────────────────────────────────────────────────────────────────

def build_T4(preds_dir: str, cfg: dict) -> Tuple[pd.DataFrame, str]:
    """T4: Stage 1 robustness under JPEG compression and Gaussian blur."""
    models = cfg["table_assignments"]["T4"]
    datasets = [("FF++", "FF++_stage1", "video"),
                ("Celeb-DF", "CelebDF", "video"),
                ("DFDC", "DFDC", "video")]
    conds = [("q95", "jpeg95"), ("q40", "jpeg40"),
             ("$\\sigma=0.5$", "blur0.5"), ("$\\sigma=3.0$", "blur3")]
    tidy, tex = [], []
    for ds_label, mkey, level in datasets:
        for cond_label, ptag in conds:
            row: dict = {"dataset": ds_label, "condition": cond_label, "level": level}
            tex_row = [ds_label, cond_label]
            for regime in models:
                df = _read_pred(preds_dir, regime, mkey, ptag)
                if df is None:
                    row[regime] = np.nan; tex_row.append("--")
                else:
                    m = _frame_metrics(df) if level == "frame" else _video_metrics(df)
                    row[regime] = m["auc"]; tex_row.append(_fmt(m["auc"]))
            tidy.append(row); tex.append(tex_row)
    latex = _latex_table(
        headers=["Dataset", "Cond."] + models,
        rows=tex,
        colspec="ll" + "r" * len(models),
        caption="Stage~1 robustness under JPEG compression and Gaussian blur.",
        label="tab:t4_stage1_robustness",
    )
    return pd.DataFrame(tidy), latex


# ── T5 ─────────────────────────────────────────────────────────────────────────

def build_T5(preds_dir: str, cfg: dict) -> Tuple[pd.DataFrame, str]:
    """T5: Stage 1 Wilcoxon signed-rank tests, Bonferroni-corrected."""
    comparisons = cfg["table_assignments"]["T5"]
    per_ds: dict = defaultdict(list)
    for c in comparisons:
        per_ds[c["dataset_label"]].append(c)

    tidy, tex = [], []
    for ds_label, group in per_ds.items():
        n_cmp = len(group)
        for c in group:
            a_df = _read_pred(preds_dir, c["regime_A"], c["manifest_key"])
            b_df = _read_pred(preds_dir, c["regime_B"], c["manifest_key"])
            if a_df is None or b_df is None:
                tidy.append(dict(c, p_corrected=np.nan, sig="No", better="--"))
                tex.append([ds_label, f"{c['regime_A']} vs {c['regime_B']}",
                             "--", "No", "--"])
                continue
            level = c.get("level", "video")
            if level == "frame":
                n = min(len(a_df), len(b_df))
                stats = wilcoxon_bonferroni(
                    a_df["score"].to_numpy()[:n], a_df["label"].to_numpy()[:n],
                    b_df["score"].to_numpy()[:n], b_df["label"].to_numpy()[:n],
                    n_comparisons=n_cmp)
            else:
                from src.eval.aggregation import align_paired_video
                a_s, a_l, b_s, b_l, _ = align_paired_video(a_df, b_df)
                stats = wilcoxon_bonferroni(a_s, a_l, b_s, b_l,
                                             n_comparisons=n_cmp)
            better = c["regime_A"] if stats["better"] == "A" else c["regime_B"]
            p_str = (r"$<0.001$" if stats["p_corrected"] < 0.001
                     else _fmt(stats["p_corrected"], 3))
            tidy.append(dict(c, p_corrected=stats["p_corrected"],
                              sig="Yes" if stats["significant"] else "No",
                              better=better, effect_d=stats["effect_d"],
                              n=stats["n"]))
            tex.append([ds_label, f"{c['regime_A']} vs {c['regime_B']}",
                        p_str, "Yes" if stats["significant"] else "No", better])
    latex = _latex_table(
        headers=["Dataset", "Comparison", "$p$ (corr.)", "Sig.", "Better"],
        rows=tex,
        colspec="llccl",
        caption="Stage~1 Wilcoxon signed-rank tests, Bonferroni-corrected.",
        label="tab:t5_stage1_wilcoxon",
    )
    return pd.DataFrame(tidy), latex


# ── T6 ─────────────────────────────────────────────────────────────────────────

def build_T6(preds_dir: str, cfg: dict) -> Tuple[pd.DataFrame, str]:
    """T6: Stage 2 five-regime cross-dataset AUC."""
    regimes = cfg["table_assignments"]["T6"]
    datasets = [("Celeb-DF", "CelebDF", "video"),
                ("DFDC",     "DFDC",    "video"),
                ("DFF",      "DFF",     "frame")]
    tidy, tex = [], []
    for regime in regimes:
        row: dict = {"regime": regime}
        tex_row = [regime]
        for ds_label, mkey, level in datasets:
            df = _read_pred(preds_dir, regime, mkey)
            if df is None:
                row[ds_label] = np.nan; tex_row.append("--")
            else:
                m = _frame_metrics(df) if level == "frame" else _video_metrics(df)
                row[ds_label] = m["auc"]; tex_row.append(_fmt(m["auc"]))
        tidy.append(row); tex.append(tex_row)
    latex = _latex_table(
        headers=["Regime"] + [d[0] for d in datasets],
        rows=tex,
        caption="Stage~2 five-regime cross-dataset AUC.",
        label="tab:t6_stage2_five_regime",
    )
    return pd.DataFrame(tidy), latex


# ── T7 ─────────────────────────────────────────────────────────────────────────

def build_T7(preds_dir: str, cfg: dict) -> Tuple[pd.DataFrame, str]:
    """T7: Stage 2 calibration — ECE and Brier score per regime."""
    regimes  = cfg["table_assignments"]["T7"]
    datasets = [("CelebDF", "celebdf"), ("DFDC", "dfdc")]
    tidy, tex = [], []
    for regime in regimes:
        row: dict = {"regime": regime}
        tex_row = [regime]
        for mkey, ds_short in datasets:
            df = _read_pred(preds_dir, regime, mkey)
            if df is None:
                row[f"{ds_short}_ece"]   = np.nan
                row[f"{ds_short}_brier"] = np.nan
                tex_row += ["--", "--"]
            else:
                m = _video_metrics(df)
                row[f"{ds_short}_ece"]   = m["ece"]
                row[f"{ds_short}_brier"] = m["brier"]
                tex_row += [_fmt(m["ece"]), _fmt(m["brier"])]
        tidy.append(row); tex.append(tex_row)
    latex = _latex_table(
        headers=["Regime", "Celeb-DF ECE", "Celeb-DF Brier",
                 "DFDC ECE", "DFDC Brier"],
        rows=tex,
        colspec="l" + "r" * 4,
        caption="Stage~2 calibration (ECE, Brier score) per regime.",
        label="tab:t7_stage2_calibration",
    )
    return pd.DataFrame(tidy), latex


# ── T8 ─────────────────────────────────────────────────────────────────────────

def build_T8(preds_dir: str, cfg: dict) -> Tuple[pd.DataFrame, str]:
    """T8: dL-quartile FF++ per-method AUC breakdown."""
    spec = cfg["table_assignments"]["T8"]
    hi_regime = spec["highdl_regime"]
    lo_regime = spec["lowdl_regime"]
    tidy, tex = [], []
    for regime, tag in [(hi_regime, "hi"), (lo_regime, "lo")]:
        df = _read_pred(preds_dir, regime, "FF++_stage1")
        if df is None:
            continue
        pm = _per_method_metrics(df, level="video")
        for _, row in pm.iterrows():
            tidy.append({"method": row["method"], "regime": tag,
                          "auc": row["auc"]})
    # Pivot to wide format
    if tidy:
        wide = pd.DataFrame(tidy).pivot(
            index="method", columns="regime", values="auc"
        ).reset_index()
        wide.columns.name = None
        wide["delta"] = wide.get("hi", np.nan) - wide.get("lo", np.nan)
        tex_rows = []
        for _, r in wide.iterrows():
            tex_rows.append([str(r["method"]),
                              _fmt(r.get("hi", np.nan)),
                              _fmt(r.get("lo", np.nan)),
                              _fmt(r.get("delta", np.nan))])
    else:
        wide = pd.DataFrame()
        tex_rows = []
    latex = _latex_table(
        headers=["Method", "HIGHDL AUC", "LOWDL AUC", "$\\Delta$"],
        rows=tex_rows,
        caption="dL-quartile FF++ per-method AUC breakdown.",
        label="tab:t8_dl_quartile_ffpp",
    )
    return wide, latex


# ── T9 ─────────────────────────────────────────────────────────────────────────

def build_T9(preds_dir: str, cfg: dict) -> Tuple[pd.DataFrame, str]:
    """T9: dL-quartile Celeb-DF aggregation strategy comparison."""
    spec = cfg["table_assignments"]["T9"]
    hi_regime = spec["highdl_regime"]
    lo_regime = spec["lowdl_regime"]
    strategies = [("mean", "mean"), ("max", "max"),
                  ("vote", "vote"), ("frame", "frame")]
    tidy, tex = [], []
    for strat_label, strat in strategies:
        hi_df = _read_pred(preds_dir, hi_regime, "CelebDF")
        lo_df = _read_pred(preds_dir, lo_regime, "CelebDF")
        if strat == "frame":
            hi_auc = _frame_metrics(hi_df)["auc"] if hi_df is not None else np.nan
            lo_auc = _frame_metrics(lo_df)["auc"] if lo_df is not None else np.nan
        else:
            hi_auc = _video_metrics(hi_df, strat)["auc"] if hi_df is not None else np.nan
            lo_auc = _video_metrics(lo_df, strat)["auc"] if lo_df is not None else np.nan
        delta = hi_auc - lo_auc if np.isfinite(hi_auc) and np.isfinite(lo_auc) else np.nan
        tidy.append({"aggregation": strat_label, "highdl_auc": hi_auc,
                     "lowdl_auc": lo_auc, "delta": delta})
        tex.append([strat_label, _fmt(hi_auc), _fmt(lo_auc), _fmt(delta)])
    latex = _latex_table(
        headers=["Aggregation", "HIGHDL AUC", "LOWDL AUC", "$\\Delta$"],
        rows=tex,
        caption="dL-quartile Celeb-DF AUC by aggregation strategy.",
        label="tab:t9_dl_quartile_celebdf",
    )
    return pd.DataFrame(tidy), latex


# ── T10 ────────────────────────────────────────────────────────────────────────

def build_T10(preds_dir: str, cfg: dict) -> Tuple[pd.DataFrame, str]:
    """T10: T-SBI pair-sampling dL distribution statistics.

    Reads tsbi_labels.csv (or a summary CSV) and reports mean dL_mean,
    dL_std, and fraction of relaxed pairs.
    """
    tsbi_csv = cfg["table_assignments"]["T10"].get("tsbi_labels_csv", "")
    if not os.path.exists(tsbi_csv):
        empty = pd.DataFrame(columns=["stat", "dL_mean", "dL_std"])
        return empty, _latex_table(
            headers=["Stat", "dL mean", "dL std"],
            rows=[["(tsbi_labels.csv not found)", "--", "--"]],
            caption="T-SBI pair-sampling statistics.",
            label="tab:t10_tsbi_pairs",
        )

    df = pd.read_csv(tsbi_csv)
    relax_col  = "illum_relaxed" if "illum_relaxed" in df.columns else None
    dLm_col    = "illum_delta_L" if "illum_delta_L" in df.columns else "dL_mean"
    dLs_col    = "illum_delta_Lstd" if "illum_delta_Lstd" in df.columns else "dL_std"

    stats_rows = [
        ("n_pairs",       len(df),                    "--"),
        ("mean_dL_mean",  float(df[dLm_col].mean()),  _fmt(float(df[dLs_col].mean()))),
        ("median_dL_mean",float(df[dLm_col].median()),_fmt(float(df[dLs_col].median()))),
        ("std_dL_mean",   float(df[dLm_col].std()),   _fmt(float(df[dLs_col].std()))),
    ]
    if relax_col:
        stats_rows.append(("frac_relaxed",
                           float(df[relax_col].mean()),
                           "--"))

    tidy = pd.DataFrame(
        [(s, v, "--") for s, v, _ in stats_rows],
        columns=["stat", "dL_mean", "dL_std"],
    )
    tex_rows = [[s, _fmt(float(v)) if isinstance(v, float) else str(v), d]
                for s, v, d in stats_rows]
    latex = _latex_table(
        headers=["Stat", "dL mean", "dL std"],
        rows=tex_rows,
        caption="T-SBI pair-sampling dL distribution statistics.",
        label="tab:t10_tsbi_pairs",
    )
    return tidy, latex


# ── public registry ────────────────────────────────────────────────────────────

TABLE_BUILDERS: Dict[str, callable] = {
    "T1":  build_T1,
    "T2":  build_T2,
    "T3":  build_T3,
    "T4":  build_T4,
    "T5":  build_T5,
    "T6":  build_T6,
    "T7":  build_T7,
    "T8":  build_T8,
    "T9":  build_T9,
    "T10": build_T10,
}


def build_all_tables(
    preds_dir: str,
    cfg: dict,
    out_dir: str,
    tables: Optional[List[str]] = None,
) -> Dict[str, Tuple[pd.DataFrame, str]]:
    """Build and persist all (or a subset of) tables.

    Writes:
        <out_dir>/tables_csv/<table_id>.csv
        <out_dir>/tables_tex/<table_id>.tex

    Returns dict {table_id: (tidy_df, latex_str)}.
    """
    import os

    os.makedirs(os.path.join(out_dir, "tables_csv"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tables_tex"), exist_ok=True)

    wanted = tables or list(TABLE_BUILDERS.keys())
    results: dict = {}
    for tid in wanted:
        if tid not in TABLE_BUILDERS:
            continue
        # Skip if the required table_assignment key is missing
        ta = cfg.get("table_assignments", {})
        if tid not in ta:
            continue
        tidy, latex = TABLE_BUILDERS[tid](preds_dir, cfg)
        tidy.to_csv(
            os.path.join(out_dir, "tables_csv", f"{tid}.csv"), index=False)
        with open(os.path.join(out_dir, "tables_tex", f"{tid}.tex"), "w") as f:
            f.write(latex + "\n")
        results[tid] = (tidy, latex)
    return results
