"""
experiments/c4_dl_quartile.py
================================
Entry point for experiment C.4 — multi-seed dL-quartile diagnostic.

Builds HIGHDL and LOWDL manifest variants from the C_pure manifests,
then trains the C_mix recipe on each variant across 3 seeds.

Steps:
  1. Build HIGHDL / LOWDL manifests (using src/tsbi/gating.py).
  2. Train EfficientNet-B4 on each variant × each seed (total 6 runs).
  3. Evaluate on all cross-dataset test sets.
  4. Write per-run metrics and a summary CSV.

CLI:
    thesis-dlquartile --config configs/stage2/c4_dl_quartile.yaml [--force]
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

import numpy as np
import pandas as pd
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="dL-quartile multi-seed diagnostic (C.4)")
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
    out_dir  = Path(out_root) / "c4_dl_quartile" / cfg_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "metrics.csv").exists() and not args.force:
        logger.info(f"Already done at {out_dir}. Use --force to rerun.")
        sys.exit(0)

    seeds           = cfg.get("seeds", [42, 123, 235])
    cpure_train_csv = cfg.get("cpure_train_csv", "")
    cpure_val_csv   = cfg.get("cpure_val_csv",   "")
    tsbi_csv        = cfg.get("tsbi_labels_csv",  "")

    for path in (cpure_train_csv, cpure_val_csv, tsbi_csv):
        if not os.path.exists(path):
            logger.error(f"Required file not found: {path}")
            logger.error("Run make manifests and make tsbi first.")
            sys.exit(1)

    # ── Step 1: Build HIGHDL / LOWDL manifests ────────────────────────────
    manifests_dir = out_dir / "manifests"
    from src.tsbi.gating import build_highdl_lowdl_manifests
    logger.info("Building HIGHDL / LOWDL manifests...")
    build_highdl_lowdl_manifests(
        tsbi_csv=tsbi_csv,
        train_manifest_csv=cpure_train_csv,
        val_manifest_csv=cpure_val_csv,
        out_dir=str(manifests_dir),
    )

    # ── Step 2: Train each variant × seed ─────────────────────────────────
    from src.train.loop import set_all_seeds
    from src.train.stage2_loop import train_stage2
    from src.models.efficientnet import build_model
    from src.eval.metrics import compute_metrics
    from src.eval.cache import DiskCache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache  = DiskCache(out_dir / "_cache")

    # Cross-dataset test manifests
    test_manifests = cfg.get("test_manifests", {
        "CelebDF": str(Path(out_root) / "crops" / "celebdf" / "manifest.csv"),
        "DFDC":    str(Path(out_root) / "crops" / "dfdc"    / "manifest.csv"),
        "DFF":     str(Path(out_root) / "crops" / "dff"     / "manifest.csv"),
    })

    all_metrics = []

    for variant in ("HIGHDL", "LOWDL"):
        train_csv = str(manifests_dir / f"C_pure_{variant}_train.csv")
        val_csv   = str(manifests_dir / f"C_pure_{variant}_val.csv")
        if not os.path.exists(train_csv):
            logger.warning(f"Manifest not found for {variant}: {train_csv}")
            continue

        with open(train_csv) as f:
            train_rows = list(csv.DictReader(f))
        with open(val_csv) as f:
            val_rows = list(csv.DictReader(f))

        for seed in seeds:
            run_dir = out_dir / f"{variant}_seed{seed}"
            if (run_dir / "best.pt").exists() and not args.force:
                logger.info(f"  CACHED  {variant}|seed={seed}")
            else:
                set_all_seeds(seed)
                model = build_model(pretrained=True)
                train_stage2(
                    model=model,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    device=device,
                    out_dir=str(run_dir),
                    epochs=cfg.get("training", {}).get("max_epochs", 20),
                    batch_size=cfg.get("training", {}).get("batch_size", 32),
                    lr=float(cfg.get("optimizer", {}).get("lr", 1e-4)),
                    weight_decay=float(cfg.get("optimizer", {}).get("weight_decay", 1e-4)),
                    seed=seed,
                )

            # Write flat alias so d_eval.py finds it at the expected path:
            # outputs/c4_dl_quartile/<variant>_seed<seed>/best.pt
            flat_dir = Path(out_root) / "c4_dl_quartile" / f"{variant}_seed{seed}"
            flat_dir.mkdir(parents=True, exist_ok=True)
            if (run_dir / "best.pt").exists():
                import shutil as _shutil
                _shutil.copy2(str(run_dir / "best.pt"), str(flat_dir / "best.pt"))

            # Evaluate
            ckpt = torch.load(str(run_dir / "best.pt"), map_location=device)
            model = build_model(pretrained=False)
            model.load_state_dict(ckpt["model"])
            model.eval().to(device)

            from torchvision import transforms
            from src.data.dataset import FaceCropDataset
            val_tf = transforms.Compose([
                transforms.Resize((380, 380)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            # FF++ uses split='test'; CelebDF/DFDC/DFF use split='all'
            for ds_key, ds_csv in test_manifests.items():
                if not os.path.exists(ds_csv):
                    continue
                eval_split = "test" if "ff" in ds_key.lower() else "all"
                ds = FaceCropDataset.from_csv(ds_csv, split=eval_split,
                                               transform=val_tf)
                loader = torch.utils.data.DataLoader(
                    ds, batch_size=32, shuffle=False, num_workers=0)
                scores_list, labels_list = [], []
                with torch.no_grad():
                    for batch in loader:
                        imgs, lbls, *_ = batch
                        probs = torch.sigmoid(
                            model(imgs.to(device)).squeeze(1)).cpu()
                        scores_list.extend(probs.tolist())
                        labels_list.extend(lbls.tolist())
                s = np.array(scores_list)
                l = np.array(labels_list, dtype=np.int32)
                cache.save_inference(f"{variant}_seed{seed}", ds_key, s, l)
                m = compute_metrics(l, s)
                m["variant"] = variant
                m["seed"]    = seed
                m["dataset"] = ds_key
                all_metrics.append(m)
                logger.info(f"  [{variant}|{ds_key}|seed={seed}]  AUC={m['auc']:.4f}")

    # ── Step 3: Write summary ─────────────────────────────────────────────
    pd.DataFrame(all_metrics).to_csv(out_dir / "metrics.csv", index=False)
    logger.info(f"Wrote {out_dir / 'metrics.csv'}")

    sidecar = {
        "config":    str(args.config),
        "cfg_hash":  cfg_hash,
        "seeds":     seeds,
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
