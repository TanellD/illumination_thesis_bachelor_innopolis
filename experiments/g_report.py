"""
experiments/g_report.py
========================
Entry point for experiment G — report layer (tables + figures + RESULTS.md).

This is the only experiment that is purely CPU-only and reads nothing but
previously-computed CSVs and NPZ caches.  It can run on a laptop without
GPU access.

CLI:
    thesis-g-report --config configs/eval/report.yaml [--tables T1 T3 T6]
                    [--figures all] [--force]

Required config keys:
    output_root:   base output directory
    preds_dir:     directory containing per-frame prediction CSVs
                   (written by inference pass of g_report or unified evaluator)
    cache_root:    path to DiskCache root for reliability diagrams
    robustness_csv: path to robustness_grid.csv from experiment E
    table_assignments: (see src/report/tables.py for schema)

Output (under <output_root>/g_report/<cfg_hash>/):
    tables_csv/   T1.csv … T10.csv
    tables_tex/   T1.tex … T10.tex
    figures/      reliability_diagrams.png, robustness_curves.png, stage1_auc_bars.png
    RESULTS.md
    sidecar.json
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report layer: tables + figures + RESULTS.md")
    p.add_argument("--config",  required=True)
    p.add_argument("--tables",  nargs="*", default=None,
                   help="Table IDs to build (default: all available). "
                        "E.g. --tables T1 T3 T6")
    p.add_argument("--figures", nargs="*", default=None,
                   help="Figure IDs to build (default: all). "
                        "E.g. --figures reliability robustness")
    p.add_argument("--force",   action="store_true",
                   help="Regenerate even if outputs already exist")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t0   = time.time()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_root_str = os.environ.get("THESIS_OUTPUT_ROOT",
                                   cfg.get("output_root", "outputs"))
    cfg["output_root"] = out_root_str

    cfg_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()[:12]
    out_dir  = Path(out_root_str) / "g_report" / cfg_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    # preds_dir: config value → resolve against out_root → auto-discover
    _preds_cfg = cfg.get("preds_dir", "")
    if _preds_cfg and not os.path.isabs(_preds_cfg):
        _preds_cfg = _preds_cfg.replace("outputs/", out_root_str.rstrip("/") + "/")
    if _preds_cfg and os.path.exists(_preds_cfg):
        preds_dir = _preds_cfg
    else:
        import glob as _glob
        _candidates = sorted(_glob.glob(
            str(Path(out_root_str) / "d_eval" / "*" / "predictions")))
        preds_dir = _candidates[-1] if _candidates else str(out_dir / "predictions")
        if _candidates:
            logger.info(f"Auto-discovered predictions dir: {preds_dir}")
    # cache_root: resolve against out_root, then auto-discover hashed d_eval cache
    _cache_cfg = cfg.get("cache_root", "")
    if _cache_cfg and not os.path.isabs(_cache_cfg):
        _cache_cfg = _cache_cfg.replace("outputs/", out_root_str.rstrip("/") + "/")
    if _cache_cfg and os.path.exists(_cache_cfg):
        cache_root = _cache_cfg
    else:
        import glob as _glob
        _cache_hits = sorted(_glob.glob(
            str(Path(out_root_str) / "d_eval" / "*" / "_cache")))
        cache_root = _cache_hits[-1] if _cache_hits else ""
        if cache_root:
            logger.info(f"Auto-discovered inference cache: {cache_root}")

    robustness_csv = cfg.get("robustness_csv", "")
    if robustness_csv and not os.path.isabs(robustness_csv):
        robustness_csv = robustness_csv.replace("outputs/",
                                                 out_root_str.rstrip("/") + "/")
    # Auto-discover if configured path doesn't exist
    if not robustness_csv or not os.path.exists(robustness_csv):
        import glob as _glob
        _rob_hits = sorted(_glob.glob(
            str(Path(out_root_str) / "e_robustness" / "*" / "robustness_grid.csv")))
        if _rob_hits:
            robustness_csv = _rob_hits[-1]
            logger.info(f"Auto-discovered robustness CSV: {robustness_csv}")
    # t3_metrics_csv: use configured path, or fall back to the T3 table just built
    metrics_csv = cfg.get("t3_metrics_csv", "")
    if not metrics_csv or not os.path.exists(metrics_csv):
        _t3_candidate = str(out_dir / "tables_csv" / "T3.csv")
        if os.path.exists(_t3_candidate):
            metrics_csv = _t3_candidate
    datasets_calib = cfg.get("calibration_datasets", ["CelebDF", "DFDC", "DFF"])

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    tables_built:  dict = {}
    figures_built: dict = {}

    # ── Tables ─────────────────────────────────────────────────────────────
    wanted_tables = args.tables or list(cfg.get("table_assignments", {}).keys())
    if wanted_tables:
        logger.info(f"Building tables: {wanted_tables}")
        from src.report.tables import build_all_tables
        results = build_all_tables(
            preds_dir=preds_dir,
            cfg=cfg,
            out_dir=str(out_dir),
            tables=wanted_tables,
        )
        for tid, (tidy, _) in results.items():
            tables_built[tid] = {
                "csv": str(out_dir / "tables_csv" / f"{tid}.csv"),
                "tex": str(out_dir / "tables_tex" / f"{tid}.tex"),
            }
            logger.info(f"  {tid}: {len(tidy)} rows")

    # ── Figures ────────────────────────────────────────────────────────────
    wanted_figs = args.figures or ["reliability", "robustness", "bars"]
    from src.report.figures import (
        plot_reliability_diagrams,
        plot_robustness_curves,
        plot_stage1_auc_bars,
    )

    if "reliability" in wanted_figs and cache_root and os.path.exists(cache_root):
        out_path = str(fig_dir / "reliability_diagrams.png")
        if not os.path.exists(out_path) or args.force:
            logger.info("Plotting reliability diagrams...")
            plot_reliability_diagrams(
                cache_root=cache_root,
                datasets=datasets_calib,
                out_path=out_path,
            )
        figures_built["reliability"] = out_path

    if "robustness" in wanted_figs and robustness_csv and os.path.exists(robustness_csv):
        out_path = str(fig_dir / "robustness_curves.png")
        if not os.path.exists(out_path) or args.force:
            logger.info("Plotting robustness curves...")
            plot_robustness_curves(
                robustness_csv=robustness_csv,
                out_path=out_path,
            )
        figures_built["robustness"] = out_path

    if "bars" in wanted_figs and metrics_csv and os.path.exists(metrics_csv):
        out_path = str(fig_dir / "stage1_auc_bars.png")
        if not os.path.exists(out_path) or args.force:
            logger.info("Plotting Stage 1 AUC bars...")
            plot_stage1_auc_bars(
                metrics_csv=metrics_csv,
                out_path=out_path,
            )
        figures_built["bars"] = out_path

    # ── RESULTS.md ─────────────────────────────────────────────────────────
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = "unknown"

    import torch
    run_info = {
        "config_hash":    cfg_hash,
        "git_sha":        git_sha,
        "timestamp":      datetime.datetime.utcnow().isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version":  torch.__version__,
        "hostname":       socket.gethostname(),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu"),
        "tables":  tables_built,
        "figures": figures_built,
        "experiments": cfg.get("experiments_metadata", []),
    }

    from src.report.manifest_doc import write_results_md
    results_md = write_results_md(str(out_dir), run_info)
    logger.info(f"Wrote {results_md}")

    # ── Sidecar ────────────────────────────────────────────────────────────
    sidecar = {**run_info, "elapsed_s": round(time.time() - t0, 1)}
    with open(out_dir / "sidecar.json", "w") as f:
        json.dump(sidecar, f, indent=2)

    logger.info(f"Done. Outputs in {out_dir}")
    logger.info(f"  Tables built:  {sorted(tables_built.keys())}")
    logger.info(f"  Figures built: {sorted(figures_built.keys())}")


if __name__ == "__main__":
    main()
