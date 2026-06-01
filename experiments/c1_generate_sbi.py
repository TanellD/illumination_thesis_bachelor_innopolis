"""
experiments/c1_generate_sbi.py
================================
Entry point for classic SBI generation (C.1 — SBI component).

Reads real face crops from a manifest CSV and produces SBI fakes
(Shiohara & Yamasaki, CVPR 2022 Self-Blended Images).

CLI:
    thesis-sbi --config configs/stage2/tsbi.yaml \
               --input-csv outputs/crops/ff++/manifest.csv \
               --out-dir   outputs/sbi \
               --out-csv   outputs/sbi_labels.csv \
               [--per-crop 1] [--splits train] [--force]

Key config keys used from tsbi.yaml:
    jpeg_quality_range:  [75, 98]
    seed:                42
    splits:              [train]
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classic SBI generation")
    p.add_argument("--config",    required=True)
    p.add_argument("--input-csv", default=None,
                   help="Real-crop manifest CSV. Overrides THESIS_FFPP_ROOT path.")
    p.add_argument("--out-dir",   default=None)
    p.add_argument("--out-csv",   default=None)
    p.add_argument("--per-crop",  type=int, default=1)
    p.add_argument("--splits",    nargs="+", default=None)
    p.add_argument("--force",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t0   = time.time()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Path resolution: CLI > env > config > defaults
    ff_root = os.environ.get("THESIS_FFPP_ROOT", cfg.get("data_dir", ""))
    out_root = os.environ.get("THESIS_OUTPUT_ROOT",
                               cfg.get("output_root", "outputs"))

    input_csv = (args.input_csv
                 or os.path.join(out_root, "crops", "ff++", "manifest.csv"))
    out_dir   = args.out_dir  or os.path.join(out_root, "sbi")
    out_csv   = args.out_csv  or os.path.join(out_root, "sbi_labels.csv")
    splits    = args.splits   or cfg.get("splits", ["train"])

    if not os.path.exists(input_csv):
        logger.error(f"Input manifest not found: {input_csv}")
        logger.error("Run 'make crop' first, or pass --input-csv explicitly.")
        sys.exit(1)

    if os.path.exists(out_csv) and not args.force:
        logger.info(f"Output CSV already exists: {out_csv}. Use --force to rerun.")
        sys.exit(0)

    from src.sbi.generator import generate_sbi_from_manifest

    qrange = tuple(cfg.get("jpeg_quality_range", [75, 98]))
    seed   = cfg.get("seed", 42)
    per_crop = args.per_crop

    logger.info(f"Generating SBI fakes from {input_csv}")
    logger.info(f"  splits={splits}  per_crop={per_crop}  q={qrange}  seed={seed}")

    n_done = generate_sbi_from_manifest(
        input_csv=input_csv,
        out_dir=out_dir,
        out_csv=out_csv,
        splits=splits,
        per_crop=per_crop,
        jpeg_quality_range=qrange,
        seed=seed,
    )
    logger.info(f"Done: {n_done} SBI crops written to {out_dir}")

    sidecar = {
        "input_csv": input_csv,
        "out_csv":   out_csv,
        "splits":    splits,
        "per_crop":  per_crop,
        "n_done":    n_done,
        "elapsed_s": round(time.time() - t0, 1),
        "hostname":  socket.gethostname(),
    }
    try:
        import subprocess
        sidecar["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        sidecar["git_sha"] = "unknown"

    sidecar_path = Path(out_dir) / "sbi_sidecar.json"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)


if __name__ == "__main__":
    main()
