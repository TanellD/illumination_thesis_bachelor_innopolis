"""
experiments/b3_context_crop.py
================================
Entry point for experiment B.3 — context-crop diagnostic.

Re-runs the noise bottleneck diagnostic with two bbox expansion factors
(1.3× tight vs 2.7× context) on FF++ test samples and reports:
  - Per-strategy AUC for L1–L4 classifiers (5-fold CV)
  - Corpus-wise SNR (between-class variance / within-class variance)

CLI:
    thesis-b3 --config configs/stage1/b3_context_crop.yaml [--force]
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
    p = argparse.ArgumentParser(description="Context-crop noise diagnostic (B.3)")
    p.add_argument("--config", required=True)
    p.add_argument("--force",  action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t0   = time.time()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_root = os.environ.get("THESIS_OUTPUT_ROOT",
                               cfg.get("output_root", "outputs"))
    cfg_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()[:12]
    out_dir  = Path(out_root) / "b3_context_crop" / cfg_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    result_csv = out_dir / "context_crop_results.csv"
    if result_csv.exists() and not args.force:
        logger.info(f"Results already exist at {result_csv}. Use --force to rerun.")
        sys.exit(0)

    manifest = cfg.get("manifest",
                       str(Path(out_root) / "crops" / "ff++" / "manifest.csv"))
    if not os.path.exists(manifest):
        logger.error(f"Manifest not found: {manifest}")
        sys.exit(1)

    noise_weights = cfg.get("noise_weights", None)
    if not noise_weights:
        # Try paths.yaml
        try:
            from src.utils.paths import load_paths
            noise_weights = str(load_paths().weights.noiseprint)
        except Exception:
            noise_weights = None

    from src.noise.context_crop import run_context_crop_diagnostic

    result_df = run_context_crop_diagnostic(
        manifest_csv=manifest,
        output_dir=str(out_dir),
        noise_weights_path=noise_weights,
        n_samples=cfg.get("n_samples_ffpp", 2000),
        split=cfg.get("split", "test"),
        tight_expansion=cfg.get("tight_expansion", 1.3),
        context_expansion=cfg.get("context_expansion", 2.7),
        seed=cfg.get("seed", 42),
    )

    logger.info("\n=== B.3 Context-crop diagnostic ===")
    for (strategy, level), row in result_df.groupby(["crop_strategy", "level"]):
        r = row.iloc[0]
        logger.info(f"  {strategy:8s} {level:20s}  "
                    f"AUC={r['mean_auc']:.4f}+-{r['std_auc']:.4f}  "
                    f"SNR={r['snr']:.4f}")

    sidecar = {
        "config":    str(args.config),
        "cfg_hash":  cfg_hash,
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
    with open(out_dir / "sidecar.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    logger.info(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
