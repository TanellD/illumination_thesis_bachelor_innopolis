"""
experiments/c3_train_regimes.py
================================
Entry point for experiment C.3 — five-regime Stage 2 training ablation.

Each regime (A, B_pure, B_mix, C_pure, C_mix) is a different training-data
composition. All other hyperparameters are identical across regimes.

CLI:
    thesis-c3-train-regimes --config configs/stage2/c3_train_regimes.yaml \
                            [--regime A] [--seed 42] [--force]

If --regime is omitted, all five regimes are run sequentially.
For the multi-seed dL-quartile diagnostic (HIGHDL / LOWDL), pass those
names as the --regime argument and point --config at the appropriate YAML.

Output (under <output_root>/c3_regimes/<cfg_hash>/<regime>_seed<seed>/):
    best.pt, last.pt, history.csv, sidecar.json

Required config keys:
    output_root:  base directory for outputs
    regimes:      dict mapping regime name → manifest paths (train_csv, val_csv)
    epochs:       20
    batch_size:   32
    lr:           1.0e-4
    weight_decay: 1.0e-4
    seeds:        [42]   (or [42, 123, 235] for multi-seed)
    device:       cuda
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)

ALL_REGIMES = ("A", "B_pure", "B_mix", "C_pure", "C_mix")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 five-regime training")
    p.add_argument("--config",  required=True)
    p.add_argument("--regime",  default=None,
                   help="Run a single regime (default: all)")
    p.add_argument("--seed",    type=int, default=None,
                   help="Override seed (default: use config seeds list)")
    p.add_argument("--force",   action="store_true")
    return p.parse_args()


def _read_manifest(csv_path: str) -> list:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = _parse_args()
    t0 = time.time()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if "THESIS_OUTPUT_ROOT" in os.environ:
        cfg["output_root"] = os.environ["THESIS_OUTPUT_ROOT"]

    cfg_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()[:12]
    out_root = Path(cfg["output_root"]) / "c3_regimes" / cfg_hash
    out_root.mkdir(parents=True, exist_ok=True)

    regimes_to_run = [args.regime] if args.regime else list(cfg.get("regimes", {}).keys())
    seeds = [args.seed] if args.seed is not None else cfg.get("seeds", [42])

    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    from src.models.efficientnet import build_model
    from src.train.stage2_loop import train_stage2
    from src.train.loop import set_all_seeds

    all_results = []

    for regime in regimes_to_run:
        if regime not in cfg.get("regimes", {}):
            logger.warning(f"Regime {regime!r} not in config — skipping")
            continue
        regime_cfg = cfg["regimes"][regime]

        for seed in seeds:
            run_dir = out_root / f"{regime}_seed{seed}"
            if (run_dir / "best.pt").exists() and not args.force:
                logger.info(f"[{regime}|seed={seed}] already done, skipping "
                            f"(use --force to rerun)")
                continue

            logger.info(f"=== Regime: {regime}  Seed: {seed} ===")
            set_all_seeds(seed)

            train_rows = _read_manifest(regime_cfg["train_csv"])
            val_rows   = _read_manifest(regime_cfg["val_csv"])
            logger.info(f"  train: {len(train_rows):,}  val: {len(val_rows):,}")

            model = build_model(pretrained=True)
            result = train_stage2(
                model=model,
                train_rows=train_rows,
                val_rows=val_rows,
                device=device,
                out_dir=str(run_dir),
                epochs=cfg.get("epochs", 20),
                batch_size=cfg.get("batch_size", 32),
                lr=cfg.get("lr", 1e-4),
                weight_decay=cfg.get("weight_decay", 1e-4),
                num_workers=cfg.get("num_workers", 4),
                seed=seed,
            )

            sidecar = {
                "config":    str(args.config),
                "cfg_hash":  cfg_hash,
                "regime":    regime,
                "seed":      seed,
                "elapsed_s": round(time.time() - t0, 1),
                "best_val_video_auc": result["best_val_video_auc"],
                "torch":     torch.__version__,
                "hostname":  socket.gethostname(),
                "gpu": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else "cpu"),
            }
            try:
                import subprocess
                sidecar["git_sha"] = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                ).decode().strip()
            except Exception:
                sidecar["git_sha"] = "unknown"

            with open(run_dir / "sidecar.json", "w") as f:
                json.dump(sidecar, f, indent=2)

            # Write history CSV
            if result["history"]:
                fields = list(result["history"][0].keys())
                with open(run_dir / "history.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    for row in result["history"]:
                        writer.writerow(row)

            all_results.append({
                "regime": regime, "seed": seed,
                "best_val_video_auc": result["best_val_video_auc"],
            })
            logger.info(f"[{regime}|seed={seed}] best video AUC = "
                        f"{result['best_val_video_auc']:.4f}")

    # Summary
    if all_results:
        logger.info("\n=== Summary ===")
        for r in all_results:
            logger.info(f"  {r['regime']:10s} seed={r['seed']}  "
                        f"AUC={r['best_val_video_auc']:.4f}")

    logger.info(f"\nDone. Total elapsed: {round(time.time() - t0, 1)}s")
    logger.info(f"Outputs in: {out_root}")


if __name__ == "__main__":
    main()
