"""
experiments/e_robustness.py
============================
Entry point for experiment E — robustness perturbation grid.

Runs inference on every (model, dataset) pair under all perturbation
conditions and writes a single long-form CSV.

CLI:
    thesis-e-robustness --config configs/eval/robustness.yaml [--force]

Required config keys:
    output_root:  base output directory
    checkpoints:  dict mapping model_name → {family, ckpt_path}
    manifests:    dict mapping dataset_name → manifest_csv_path
    device:       cuda | cpu
    batch_size:   32
    num_workers:  4

Output (under <output_root>/e_robustness/<cfg_hash>/):
    robustness_grid.csv   — long-form (model, dataset, family, param, metrics)
    sidecar.json
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness perturbation grid")
    p.add_argument("--config",  required=True)
    p.add_argument("--force",   action="store_true")
    return p.parse_args()


def _load_model(family: str, ckpt_path: str, device: torch.device):
    """Load a model by family name and checkpoint path."""
    if family in ("stage1_rgb", "stage1_fusion", "stage1_residual"):
        from src.models.rgb_only      import RGBOnlyModel
        from src.models.late_fusion   import LateFusionModel
        from src.models.residual_only import ResidualOnlyModel
        cls = {"stage1_rgb": RGBOnlyModel,
               "stage1_fusion": LateFusionModel,
               "stage1_residual": ResidualOnlyModel}[family]
        model = cls()
    elif family == "stage2":
        from src.models.efficientnet import EfficientNetB4Model
        model = EfficientNetB4Model(pretrained=False)
    else:
        raise ValueError(f"Unknown model family: {family!r}")

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


def _make_loader_factory(manifest_csv: str, batch_size: int, num_workers: int,
                          target_size: int):
    """Return a factory that creates a fresh DataLoader each call."""
    from torchvision import transforms
    from src.data.dataset import FaceCropDataset
    tf = transforms.Compose([
        transforms.Resize((int(target_size * 1.1), int(target_size * 1.1))),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    def factory():
        ds = FaceCropDataset.from_csv(manifest_csv, split="test", transform=tf)
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    return factory


def main() -> None:
    args = _parse_args()
    t0   = time.time()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if "THESIS_OUTPUT_ROOT" in os.environ:
        cfg["output_root"] = os.environ["THESIS_OUTPUT_ROOT"]

    cfg_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()[:12]
    out_dir  = Path(cfg["output_root"]) / "e_robustness" / cfg_hash
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv  = out_dir / "robustness_grid.csv"

    if out_csv.exists() and not args.force:
        logger.info(f"Output already exists at {out_csv}. Use --force to rerun.")
        sys.exit(0)

    device     = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = cfg.get("batch_size", 32)
    n_workers  = cfg.get("num_workers", 4)

    from src.robustness.grid import run_robustness_grid

    all_rows: list = []
    for model_name, model_cfg in cfg["checkpoints"].items():
        family    = model_cfg["family"]
        ckpt_path = model_cfg["ckpt"]
        target_sz = model_cfg.get("target_size", 380 if family == "stage2" else 299)

        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path} — skipping {model_name}")
            continue

        logger.info(f"Loading {model_name} from {ckpt_path}")
        model = _load_model(family, ckpt_path, device)

        for dataset_name, manifest_csv in cfg["manifests"].items():
            if not os.path.exists(manifest_csv):
                logger.warning(f"Manifest not found: {manifest_csv} — skipping")
                continue
            logger.info(f"  Evaluating {model_name} on {dataset_name} ...")
            loader_factory = _make_loader_factory(
                manifest_csv, batch_size, n_workers, target_sz)

            partial_csv = str(out_dir / f"{model_name}__{dataset_name}.csv")
            df = run_robustness_grid(
                model=model,
                loader_factory=loader_factory,
                model_name=model_name,
                dataset_name=dataset_name,
                output_csv=partial_csv,
                device=device,
            )
            all_rows.append(df)

    if all_rows:
        import pandas as pd
        full_df = pd.concat(all_rows, ignore_index=True)
        full_df.to_csv(str(out_csv), index=False)
        logger.info(f"Wrote {out_csv} ({len(full_df)} rows)")

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
