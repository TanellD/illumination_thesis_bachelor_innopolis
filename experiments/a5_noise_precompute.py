"""
A.5 — Noiseprint++ noise-map precomputation.

Runs on the manifest produced by A.2 (a2_crop.py).  Adds noise_crop_path and
noise_full_path columns to a new manifest CSV in the noise output directory.

Usage:
  thesis-noise-pre --config configs/data/ffpp.yaml
  thesis-noise-pre --config configs/data/celebdf.yaml [--split test] [--force]
  python experiments/a5_noise_precompute.py --config configs/data/ffpp.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time

import yaml

from src.noise.precompute import run_precompute

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _write_sidecar(manifest_path: str, cfg: dict, elapsed: float) -> None:
    import platform
    import torch
    sidecar = {
        "config":      cfg,
        "config_hash": hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()
        ).hexdigest()[:12],
        "elapsed_s":   round(elapsed, 1),
        "python":      sys.version,
        "torch":       torch.__version__,
        "hostname":    platform.node(),
        "gpu":         torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    sidecar_path = manifest_path.replace(".csv", ".sidecar.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    logger.info("Sidecar written: %s", sidecar_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A.5 Noiseprint++ precomputation")
    p.add_argument("--config", required=True,
                   help="Dataset config YAML (same one used for A.2)")
    p.add_argument("--manifest", default=None,
                   help="Override manifest CSV path (default: outputs/crops/<dataset>/manifest.csv)")
    p.add_argument("--output-dir", default=None,
                   help="Override noise output directory")
    p.add_argument("--split", default=None,
                   help="Process only this split (e.g. 'test'). Default: all splits.")
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if .pt files already exist")
    p.add_argument("--tiny", action="store_true",
                   help="Tiny mode: process at most 32 rows (for smoke tests)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset    = cfg.get("dataset", "")
    output_root = os.environ.get("THESIS_OUTPUT_ROOT", cfg.get("output_dir", "outputs"))

    # Manifest from A.2 output
    manifest = args.manifest or os.path.join(output_root, "crops", dataset, "manifest.csv")
    if not os.path.exists(manifest):
        logger.error("Manifest not found: %s  — run a2_crop.py first.", manifest)
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(output_root, "noise", dataset)

    weights_path = cfg.get("weights", {}).get("noiseprint") or os.environ.get(
        "THESIS_NOISEPRINT_WEIGHTS",
        "artifacts/models_weights/noiseprintplusplus_weights.pth",
    )

    # In tiny mode, trim the manifest to 32 rows for a fast smoke test
    if args.tiny:
        import pandas as pd
        df = pd.read_csv(manifest).head(32)
        tiny_manifest = os.path.join(output_dir, "manifest_tiny.csv")
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(tiny_manifest, index=False)
        manifest = tiny_manifest
        logger.info("Tiny mode: processing %d rows", len(df))

    t0 = time.time()
    out_manifest = run_precompute(
        manifest_csv=manifest,
        output_dir=output_dir,
        weights_path=weights_path,
        crop_size=cfg.get("noise_crop_size", 299),
        split_filter=args.split,
        force=args.force,
    )
    elapsed = time.time() - t0

    _write_sidecar(out_manifest, cfg, elapsed)
    logger.info("Done in %.1fs. Noise manifest: %s", elapsed, out_manifest)


if __name__ == "__main__":
    main()
