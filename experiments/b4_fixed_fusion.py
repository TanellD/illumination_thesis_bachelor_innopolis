"""
experiments/b4_fixed_fusion.py
================================
Entry point for experiment B.4 — fixed-fusion variants.

Trains StatNoise-Fusion and ResAware-Fusion with 5 seeds each on the
Stage 1 small FF++ corpus, then evaluates cross-dataset on Celeb-DF.

Both models use the same training protocol as B.1 (§3.6):
  - AdamW with per-component LRs (see src/train/loop.py build_optimizer)
  - 3-epoch warmup, ReduceLROnPlateau on val macro-F1
  - Early stopping patience=7, gradient clipping at 1.0

CLI:
    thesis-b4 --config configs/stage1/b4_fixed_fusion.yaml [--force]
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
    p = argparse.ArgumentParser(description="Stage 1 fixed-fusion variants (B.4)")
    p.add_argument("--config", required=True)
    p.add_argument("--force",  action="store_true")
    return p.parse_args()


def _build_model(model_name: str, noise_weights_path):
    """Instantiate a Stage 1 fixed-fusion model by thesis name."""
    noise_model = None
    if noise_weights_path and os.path.exists(str(noise_weights_path)):
        from src.models.noiseprintpp import TruForNoiseModel
        noise_model = TruForNoiseModel(weights_path=str(noise_weights_path))

    if model_name == "StatNoise-Fusion":
        from src.models.statnoise_fusion import StatNoiseFusionModel
        return StatNoiseFusionModel()

    if model_name == "ResAware-Fusion":
        from src.models.resaware_fusion import ResAwareFusionModel
        return ResAwareFusionModel()

    raise ValueError(f"Unknown fixed-fusion model: {model_name!r}. "
                     f"Expected: StatNoise-Fusion, ResAware-Fusion")


def main() -> None:
    args = _parse_args()
    t0   = time.time()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_root = os.environ.get("THESIS_OUTPUT_ROOT",
                               cfg.get("output_root", "outputs"))
    cfg_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()[:12]
    out_dir  = Path(out_root) / "b4_fixed_fusion" / cfg_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "metrics.csv").exists() and not args.force:
        logger.info(f"Already done at {out_dir}. Use --force to rerun.")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds  = cfg.get("seeds", [42, 123, 456, 789, 1337])
    models = cfg.get("models", ["StatNoise-Fusion", "ResAware-Fusion"])

    noise_weights = cfg.get("noise", {}).get("weights", None)
    if not noise_weights:
        try:
            from src.utils.paths import load_paths
            noise_weights = str(load_paths().weights.noiseprint)
        except Exception:
            noise_weights = None

    from src.train.loop import set_all_seeds, train_model
    from src.eval.metrics import compute_metrics
    from src.eval.cache import DiskCache
    from torchvision import transforms
    from src.data.dataset import NoiseCropDataset, FaceCropDataset

    cache = DiskCache(out_dir / "_cache")
    all_metrics = []

    ff_manifest = cfg.get("manifests", {}).get(
        "ff++", str(Path(out_root) / "crops" / "ff++" / "manifest.csv"))
    cdf_manifest = cfg.get("manifests", {}).get(
        "celebdf", str(Path(out_root) / "crops" / "celebdf" / "manifest.csv"))

    val_tf = transforms.Compose([
        transforms.Resize((330, 330)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    for model_name in models:
        for seed in seeds:
            set_all_seeds(seed)
            model = _build_model(model_name, noise_weights).to(device)

            needs_noise = True  # both fixed-fusion variants need noise crops
            train_ds = NoiseCropDataset.from_csv(ff_manifest, split="train",
                                                   transform=val_tf)
            val_ds   = NoiseCropDataset.from_csv(ff_manifest, split="val",
                                                   transform=val_tf)

            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=cfg.get("training", {}).get("batch_size", 32),
                shuffle=True, num_workers=0, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=32, shuffle=False, num_workers=0)

            logger.info(f"Training {model_name}, seed={seed}")
            model, _, _ = train_model(
                model, train_loader, val_loader, device,
                model_name=model_name, seed=seed,
                epochs=cfg.get("training", {}).get("max_epochs", 30),
                lr=float(cfg.get("optimizer", {}).get("lr", 1e-4)),
            )

            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(),
                       ckpt_dir / f"{model_name}_seed{seed}.pt")

            # Evaluate on FF++ and Celeb-DF
            for ds_key, ds_csv in [("FF++_stage1", ff_manifest),
                                     ("CelebDF",     cdf_manifest)]:
                if not os.path.exists(ds_csv):
                    continue
                test_ds = NoiseCropDataset.from_csv(ds_csv, split="test",
                                                      transform=val_tf)
                loader  = torch.utils.data.DataLoader(
                    test_ds, batch_size=32, shuffle=False, num_workers=0)
                model.eval()
                scores_list, labels_list = [], []
                with torch.no_grad():
                    for batch in loader:
                        rgb, noise, lbls, _ = batch
                        probs = torch.softmax(
                            model(rgb.to(device), noise.to(device)), 1)[:, 1]
                        scores_list.extend(probs.cpu().tolist())
                        labels_list.extend(lbls.tolist())
                s = np.array(scores_list)
                l = np.array(labels_list, dtype=np.int32)
                cache.save_inference(f"{model_name}_seed{seed}", ds_key, s, l)
                m = compute_metrics(l, s)
                m["model"] = model_name
                m["dataset"] = ds_key
                m["seed"] = seed
                all_metrics.append(m)
                logger.info(f"  [{model_name}|{ds_key}|seed={seed}]  AUC={m['auc']:.4f}")

    pd.DataFrame(all_metrics).to_csv(out_dir / "metrics.csv", index=False)
    logger.info(f"Wrote {out_dir / 'metrics.csv'}")

    sidecar = {
        "config":    str(args.config),
        "cfg_hash":  cfg_hash,
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
