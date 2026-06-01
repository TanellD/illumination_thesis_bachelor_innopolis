"""
A.2 — MTCNN face extraction for a single dataset.

Usage:
  thesis-crop --config configs/data/ffpp.yaml
  thesis-crop --config configs/data/celebdf.yaml [--force]
  python experiments/a2_crop.py --config configs/data/dfdc.yaml
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

from src.data.extract_mtcnn import extract
from src.data.manifest import validate_manifest
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _config_hash(cfg: dict) -> str:
    """Short content hash of the config dict for the sidecar JSON."""
    raw = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _write_sidecar(manifest_path: str, cfg: dict, elapsed: float) -> None:
    import platform
    import torch
    sidecar = {
        "config":       cfg,
        "config_hash":  _config_hash(cfg),
        "elapsed_s":    round(elapsed, 1),
        "python":       sys.version,
        "torch":        torch.__version__,
        "hostname":     platform.node(),
        "gpu":          torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    sidecar_path = manifest_path.replace(".csv", ".sidecar.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    logger.info("Sidecar written: %s", sidecar_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A.2 MTCNN face extraction")
    p.add_argument("--config", required=True,
                   help="Path to dataset config YAML (e.g. configs/data/ffpp.yaml)")
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if manifest already exists")
    p.add_argument("--validate", action="store_true",
                   help="Run file-existence validation after extraction")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Allow environment-variable overrides for data paths
    dataset = cfg.get("dataset", "")
    env_map = {
        "ff++":    "THESIS_FFPP_ROOT",
        "celebdf": "THESIS_CELEBDF_ROOT",
        "dfdc":    "THESIS_DFDC_ROOT",
        "dff":     "THESIS_DFF_ROOT",
    }
    if dataset in env_map:
        env_val = os.environ.get(env_map[dataset])
        if env_val:
            logger.info("Overriding data_dir from env %s=%s", env_map[dataset], env_val)
            cfg["data_dir"] = env_val

    output_dir = os.environ.get("THESIS_OUTPUT_ROOT", cfg.get("output_dir", "outputs"))
    cfg["output_dir"] = os.path.join(output_dir, "crops", dataset)

    manifest_path = os.path.join(cfg["output_dir"], "manifest.csv")
    if os.path.exists(manifest_path) and not args.force:
        logger.info("Manifest already exists: %s  (use --force to re-extract)", manifest_path)
        return

    t0 = time.time()
    manifest_path = extract(cfg)
    elapsed = time.time() - t0

    if args.validate:
        logger.info("Validating manifest …")
        df = pd.read_csv(manifest_path)
        validate_manifest(df, check_files_exist=True)
        logger.info("Validation passed.")

    _write_sidecar(manifest_path, cfg, elapsed)
    logger.info("Done in %.1fs. Manifest: %s", elapsed, manifest_path)


if __name__ == "__main__":
    main()
