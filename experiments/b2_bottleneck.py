"""
experiments/b2_bottleneck.py
=============================
Entry point for experiment B.2 — 7-level noise bottleneck diagnostic.

CLI:
    thesis-b2-bottleneck --config configs/stage1/b2_bottleneck.yaml [--force]

Required config keys:
    manifest:    path to FF++ test manifest with noise_crop_path column
    output_root: base output directory
    n_samples:   number of samples for CV (default 3000)
    split:       manifest split to use (default "test")

Output (under <output_root>/b2_bottleneck/<config_hash>/):
    bottleneck_results.csv  — L1–L7 mean AUC ± std
    sidecar.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

import yaml
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="7-level noise bottleneck diagnostic")
    p.add_argument("--config", required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t0 = time.time()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()[:12]
    if "THESIS_OUTPUT_ROOT" in os.environ:
        cfg["output_root"] = os.environ["THESIS_OUTPUT_ROOT"]

    out_dir = Path(cfg["output_root"]) / "b2_bottleneck" / cfg_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "bottleneck_results.csv").exists() and not args.force:
        logger.info(f"Output already exists at {out_dir}. Use --force to rerun.")
        sys.exit(0)

    manifest = cfg["manifest"]
    if "THESIS_FFPP_ROOT" in os.environ:
        manifest = manifest.replace(
            "/data/faceforensics", os.environ["THESIS_FFPP_ROOT"])

    from src.noise.bottleneck import run_bottleneck_diagnostic

    result_df = run_bottleneck_diagnostic(
        manifest_csv=manifest,
        output_dir=str(out_dir),
        n_samples=cfg.get("n_samples", 3000),
        split=cfg.get("split", "test"),
        seed=cfg.get("seed", 42),
    )

    logger.info("\nBottleneck diagnostic summary:")
    for _, row in result_df.iterrows():
        logger.info(
            f"  L{int(row['level'])} {row['name']:30s}  "
            f"AUC={row['mean_auc']:.4f} ± {row['std_auc']:.4f}"
        )

    sidecar = {
        "config":      str(args.config),
        "config_hash": cfg_hash,
        "manifest":    manifest,
        "n_samples":   cfg.get("n_samples", 3000),
        "elapsed_s":   round(time.time() - t0, 1),
        "torch":       torch.__version__,
        "hostname":    socket.gethostname(),
    }
    try:
        import subprocess
        sidecar["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        sidecar["git_sha"] = "unknown"

    with open(out_dir / "sidecar.json", "w") as f:
        json.dump(sidecar, f, indent=2)

    logger.info(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
