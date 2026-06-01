"""
experiments/b1_ablation.py
==========================
Entry point for experiment B.1 — Stage 1 three-model ablation.

Trains RGB-Only, Residual-Only, and Late-Fusion models on FF++ (small corpus)
across 5 seeds and evaluates cross-dataset.

CLI:
    thesis-b1-ablation --config configs/stage1/b1_ablation.yaml [--force]

Output (under <output_root>/b1_ablation/<config_hash>/):
    checkpoints/   — best_<model>_seed<seed>.pt
    preds/         — per-frame predictions CSV per (model, dataset, seed)
    metrics.csv    — aggregated mean ± std over seeds
    sidecar.json   — config hash, seeds, git SHA, versions, wall-clock time
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
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
    p = argparse.ArgumentParser(description="Stage 1 three-model ablation")
    p.add_argument("--config", required=True,
                   help="Path to YAML config file")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


def _config_hash(cfg_path: str) -> str:
    return hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest()[:12]


def _resolve_paths(cfg: dict) -> dict:
    """Override data_dir from environment variables."""
    env_map = {
        "ff++":    "THESIS_FFPP_ROOT",
        "celebdf": "THESIS_CELEBDF_ROOT",
        "dfdc":    "THESIS_DFDC_ROOT",
        "dff":     "THESIS_DFF_ROOT",
    }
    for key, env_var in env_map.items():
        if env_var in os.environ and key in cfg.get("manifests", {}):
            # Allow manifest paths to be overridden
            pass
    if "THESIS_OUTPUT_ROOT" in os.environ:
        cfg["output_root"] = os.environ["THESIS_OUTPUT_ROOT"]
    return cfg


def _build_model(model_name: str, noise_model=None):
    """Instantiate a Stage 1 model by thesis name."""
    from src.models.rgb_only      import RGBOnlyModel
    from src.models.residual_only import ResidualOnlyModel
    from src.models.late_fusion   import LateFusionModel

    if model_name == "RGB-Only":
        return RGBOnlyModel()
    if model_name == "Residual-Only":
        return ResidualOnlyModel(noise_model=noise_model)
    if model_name == "Late-Fusion":
        return LateFusionModel(noise_model=noise_model)
    raise ValueError(f"Unknown model: {model_name!r}. "
                     f"Expected one of: RGB-Only, Residual-Only, Late-Fusion")


def _build_dataset(manifest_csv: str, split: str, model_name: str, cfg: dict):
    """Return a DataLoader for the given split."""
    from torchvision import transforms
    from src.data.dataset import FaceCropDataset, NoiseCropDataset

    needs_noise = model_name in ("Residual-Only", "Late-Fusion",
                                  "StatNoise-Fusion", "ResAware-Fusion")
    is_train    = split == "train"
    target_size = cfg.get("target_size", 299)

    transform = transforms.Compose([
        transforms.Resize((330, 330)) if target_size == 299 else
            transforms.Resize((int(target_size * 1.1), int(target_size * 1.1))),
        transforms.RandomCrop(target_size) if is_train else
            transforms.CenterCrop(target_size),
        *([transforms.RandomHorizontalFlip(0.5),
           transforms.ColorJitter(brightness=0.2, contrast=0.2)]
          if is_train else []),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    if needs_noise:
        ds = NoiseCropDataset.from_csv(manifest_csv, split=split,
                                       transform=transform)
    else:
        ds = FaceCropDataset.from_csv(manifest_csv, split=split,
                                      transform=transform)

    batch_size = cfg.get("batch_size", 32)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=is_train,
        num_workers=cfg.get("num_workers", 4), pin_memory=True,
    )


def main() -> None:
    args = _parse_args()
    t0 = time.time()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_paths(cfg)
    cfg_hash = _config_hash(args.config)

    out_root = Path(cfg["output_root"]) / "b1_ablation" / cfg_hash
    out_root.mkdir(parents=True, exist_ok=True)

    if (out_root / "metrics.csv").exists() and not args.force:
        logger.info(f"Output already exists at {out_root}. Use --force to rerun.")
        sys.exit(0)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seeds  = cfg.get("seeds", [42, 123, 456, 789, 1337])
    models = cfg.get("models", ["RGB-Only", "Residual-Only", "Late-Fusion"])

    noise_weights = cfg.get("weights", {}).get("noiseprint", None)
    noise_model   = None
    if noise_weights and os.path.exists(str(noise_weights)):
        from src.models.noiseprintpp import TruForNoiseModel
        noise_model = TruForNoiseModel(weights_path=noise_weights, device=device)

    all_metrics: list = []

    for model_name in models:
        for seed in seeds:
            from src.train.loop import set_all_seeds, train_model
            set_all_seeds(seed)

            model = _build_model(model_name, noise_model=noise_model).to(device)

            train_loader = _build_dataset(cfg["manifests"]["ff++"], "train",
                                          model_name, cfg)
            val_loader   = _build_dataset(cfg["manifests"]["ff++"], "val",
                                          model_name, cfg)

            logger.info(f"Training {model_name}, seed={seed}")
            model, train_hist, val_hist = train_model(
                model, train_loader, val_loader, device,
                model_name=model_name, seed=seed,
                epochs=cfg.get("max_epochs", 30),
                lr=cfg.get("lr", 1e-3),
            )

            # Save checkpoint
            ckpt_dir = out_root / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(),
                       ckpt_dir / f"best_{model_name}_seed{seed}.pt")

            # Evaluate
            from src.eval.metrics import compute_metrics
            from src.eval.cache import DiskCache

            cache = DiskCache(out_root / "_cache")
            for ds_key, ds_csv in cfg.get("manifests", {}).items():
                if not os.path.exists(ds_csv):
                    continue
                test_loader = _build_dataset(ds_csv, "test",
                                              model_name, cfg)
                model.eval()
                scores_list: list = []
                labels_list: list = []
                vids_list:   list = []
                srcs_list:   list = []

                with torch.no_grad():
                    for batch in test_loader:
                        if len(batch) == 3:
                            imgs, lbls, _ = batch
                            probs = torch.softmax(model(imgs.to(device)), 1)[:, 1]
                        else:
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
                logger.info(f"  [{model_name}|{ds_key}|seed={seed}]  "
                            f"AUC={m['auc']:.4f}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_root / "metrics.csv", index=False)
    logger.info(f"Wrote {out_root / 'metrics.csv'}")

    sidecar = {
        "config":       str(args.config),
        "config_hash":  cfg_hash,
        "seeds":        seeds,
        "models":       models,
        "elapsed_s":    round(time.time() - t0, 1),
        "python":       sys.version,
        "torch":        torch.__version__,
        "hostname":     socket.gethostname(),
        "gpu":          (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else "cpu"),
    }
    try:
        import subprocess
        sidecar["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        sidecar["git_sha"] = "unknown"

    with open(out_root / "sidecar.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    logger.info(f"Done. Outputs in {out_root}")


if __name__ == "__main__":
    main()
