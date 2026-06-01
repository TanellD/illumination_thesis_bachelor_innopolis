"""
experiments/f_taxonomy.py
==========================
Entry point for experiment F — failure taxonomy.

Step 1 (F.1): Compute per-frame attributes for every face crop in every
              evaluation dataset.
Step 2 (F.2): Quantile-bin continuous attributes within each dataset.
Step 3 (F.3): Chi-square + Cramér's V contingency tests.
Step 4 (F.4): Cross-model failure Jaccard agreement.

Outputs written under <output_root>/f_taxonomy/<cfg_hash>/:
    attributes.parquet          — per-crop attributes
    taxonomy_summary.csv        — per-bin failure rates
    chi2_significance.csv       — chi-square + Cramér's V per (model, dataset, axis)
    failure_jaccard.csv         — pairwise Jaccard + lift
    agreement_distribution.csv  — histogram of failures-per-sample
    unique_failures.csv         — per-sample which models failed
    sidecar.json

CLI:
    thesis-f-taxonomy --config configs/eval/taxonomy.yaml [--force] [--attrs-only]

--attrs-only: only run Step 1 (useful on GPU machines where inference is done but
              the attribute computation hasn't been run yet).

Required config keys:
    output_root:     base output directory
    manifests:       dict mapping dataset_name → manifest_csv_path
    inference_cache: path to DiskCache root containing inference NPZs
    models:          list of model names matching the DiskCache keys
    attribute_axes:  list of attribute column names to test (e.g. [blur_bin, pose_bin])
    threshold:       decision threshold (default 0.5)
    num_workers:     number of parallel workers for attribute computation
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

import pandas as pd
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Failure taxonomy pipeline")
    p.add_argument("--config",     required=True)
    p.add_argument("--force",      action="store_true")
    p.add_argument("--attrs-only", action="store_true",
                   help="Only compute attributes; skip statistical tests")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t0   = time.time()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if "THESIS_OUTPUT_ROOT" in os.environ:
        cfg["output_root"] = os.environ["THESIS_OUTPUT_ROOT"]

    cfg_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()[:12]
    out_dir  = Path(cfg["output_root"]) / "f_taxonomy" / cfg_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    attrs_parquet = out_dir / "attributes.parquet"

    # ── F.1  Attribute computation ─────────────────────────────────────────
    if not attrs_parquet.exists() or args.force:
        from src.taxonomy.attributes import compute_attributes_batch

        all_paths: list = []
        for dataset_name, manifest_csv in cfg["manifests"].items():
            if not os.path.exists(manifest_csv):
                logger.warning(f"Manifest not found: {manifest_csv}")
                continue
            df_m = pd.read_csv(manifest_csv)
            col  = next((c for c in ("face_crop_path", "image_path", "filepath")
                         if c in df_m.columns), None)
            if col is None:
                logger.warning(f"No path column in {manifest_csv}")
                continue
            all_paths.extend(df_m[col].dropna().tolist())

        all_paths = list(dict.fromkeys(all_paths))   # dedup, preserve order
        logger.info(f"Computing attributes for {len(all_paths):,} crops...")
        compute_attributes_batch(
            paths=all_paths,
            output_parquet=str(attrs_parquet),
            checkpoint_every=cfg.get("checkpoint_every", 5000),
            resume=not args.force,
        )
        logger.info(f"Attributes written to {attrs_parquet}")
    else:
        logger.info(f"Attributes already exist at {attrs_parquet} — skipping")

    if args.attrs_only:
        logger.info("--attrs-only: stopping after attribute computation.")
        sys.exit(0)

    # ── Build long-form predictions DataFrame ──────────────────────────────
    from src.eval.cache import DiskCache

    cache_root = cfg.get("inference_cache")
    if not cache_root or not os.path.exists(cache_root):
        logger.error(f"inference_cache not found: {cache_root}")
        sys.exit(1)

    cache   = DiskCache(cache_root)
    models  = cfg.get("models", [])
    threshold = cfg.get("threshold", 0.5)
    attr_axes = cfg.get("attribute_axes",
                        ["blur_bin", "pose_bin", "illum_bin",
                         "eye_state", "crop_tightness", "touches_edge"])

    long_rows: list = []
    for dataset_name in cfg["manifests"]:
        for model_name in models:
            result = cache.load_inference(model_name, dataset_name)
            if result is None:
                logger.warning(f"No cache for ({model_name}, {dataset_name})")
                continue
            scores    = result["scores"].astype(float)
            labels    = result["labels"].astype(int)
            paths_arr = result.get("paths", [None] * len(scores))
            for i in range(len(scores)):
                long_rows.append({
                    "model":          model_name,
                    "dataset":        dataset_name,
                    "face_crop_path": str(paths_arr[i]) if paths_arr[i] is not None else "",
                    "score":          float(scores[i]),
                    "label":          int(labels[i]),
                })

    if not long_rows:
        logger.error("No inference results found. Run inference first.")
        sys.exit(1)

    long_df = pd.DataFrame(long_rows)

    # Join attributes
    attrs_df = pd.read_parquet(str(attrs_parquet))
    if "face_crop_path" in attrs_df.columns:
        long_df = long_df.merge(attrs_df, on="face_crop_path", how="left")
    else:
        logger.warning("attributes.parquet has no face_crop_path column")

    # ── F.2  Quantile binning ──────────────────────────────────────────────
    from src.taxonomy.contingency import (
        quantile_bin_within_dataset,
        chi_square_cramers,
        per_bin_failure_rates,
        failure_jaccard,
        agreement_distribution,
        unique_failures_table,
    )

    for cont_col in ["blur_var_lap", "L_mean", "L_std", "iris_offset_norm"]:
        if cont_col in long_df.columns:
            long_df = quantile_bin_within_dataset(long_df, cont_col)

    # ── F.3  Chi-square + Cramér's V ──────────────────────────────────────
    chi2_df = chi_square_cramers(long_df, attr_axes, threshold=threshold)
    chi2_df.to_csv(out_dir / "chi2_significance.csv", index=False)
    logger.info(f"Wrote chi2_significance.csv ({len(chi2_df)} rows)")

    summary_df = per_bin_failure_rates(long_df, attr_axes, threshold=threshold)
    summary_df.to_csv(out_dir / "taxonomy_summary.csv", index=False)
    logger.info(f"Wrote taxonomy_summary.csv ({len(summary_df)} rows)")

    # ── F.4  Jaccard agreement ────────────────────────────────────────────
    jaccard_df = failure_jaccard(long_df, threshold=threshold)
    jaccard_df.to_csv(out_dir / "failure_jaccard.csv", index=False)
    logger.info(f"Wrote failure_jaccard.csv ({len(jaccard_df)} rows)")

    agree_df = agreement_distribution(long_df, threshold=threshold)
    agree_df.to_csv(out_dir / "agreement_distribution.csv", index=False)
    logger.info(f"Wrote agreement_distribution.csv ({len(agree_df)} rows)")

    unique_df = unique_failures_table(long_df, threshold=threshold)
    unique_df.to_csv(out_dir / "unique_failures.csv", index=False)
    logger.info(f"Wrote unique_failures.csv ({len(unique_df)} rows)")

    # ── Sidecar ────────────────────────────────────────────────────────────
    sidecar = {
        "config":    str(args.config),
        "cfg_hash":  cfg_hash,
        "n_crops":   len(long_df["face_crop_path"].unique()),
        "elapsed_s": round(time.time() - t0, 1),
        "torch":     torch.__version__,
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
