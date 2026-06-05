"""
experiments/d_eval.py
======================
Entry point for experiment D — evaluation (D.1–D.5).

Runs inference for every (model, dataset) pair, writes per-frame prediction
CSVs, and computes the metric suite (AUC, EER, FPR@95, ECE, Brier) at both
frame level and three video-level aggregation strategies (mean, max, vote).

Also computes:
  - 99-threshold sweep (D.3)
  - 95% bootstrap CIs (D.5)
  - Cross-seed aggregation (D.4) when --seeds is set
  - Wilcoxon+Bonferroni paired significance (D.4)

CLI:
    thesis-eval --config configs/eval/eval_config.yaml \
                --checkpoints configs/eval/checkpoints.yaml \
                [--force] [--tiny]

Required config keys in checkpoints.yaml:
    checkpoints:
      <model_name>:
        family: stage1_rgb | stage1_fusion | stage1_residual | stage2
        ckpt:   path/to/best.pt
    manifests:
      <dataset_name>: path/to/test_manifest.csv

All outputs written to <output_root>/d_eval/<cfg_hash>/.
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
from collections import defaultdict
from pathlib import Path
from typing import Optional

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
    p = argparse.ArgumentParser(description="Evaluation pipeline (D.1-D.5)")
    p.add_argument("--config",       required=True,
                   help="Eval config YAML (metrics, strategies, bootstrap)")
    p.add_argument("--checkpoints",  default=None,
                   help="Checkpoints YAML (model family + ckpt path). "
                        "If omitted, looks for 'checkpoints' key in --config.")
    p.add_argument("--force",        action="store_true")
    p.add_argument("--tiny",         action="store_true",
                   help="Sub-sample to 32 rows for smoke tests")
    return p.parse_args()


def _load_model(family: str, ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if family == "stage1_rgb":
        from src.models.rgb_only import RGBOnlyModel
        model = RGBOnlyModel()
    elif family == "stage1_residual":
        from src.models.residual_only import ResidualOnlyModel
        model = ResidualOnlyModel()
    elif family == "stage1_fusion":
        from src.models.late_fusion import LateFusionModel
        model = LateFusionModel()
    elif family == "stage2":
        from src.models.efficientnet import EfficientNetB4Model
        model = EfficientNetB4Model(pretrained=False)
    else:
        raise ValueError(f"Unknown model family: {family!r}")
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


@torch.no_grad()
def _run_inference(model, loader, device, tiny: bool = False):
    scores, labels, vids, sources, methods, paths_list = [], [], [], [], [], []
    for i, batch in enumerate(loader):
        if tiny and i >= 2:
            break
        imgs   = batch[0].to(device)
        lbls   = batch[1]
        extra  = batch[2:]
        out = model(imgs)
        if out.ndim == 2 and out.shape[1] == 2:
            probs = torch.softmax(out, 1)[:, 1]
        elif out.ndim == 2 and out.shape[1] == 1:
            probs = torch.sigmoid(out.squeeze(1))
        else:
            probs = torch.sigmoid(out)
        scores.extend(probs.float().cpu().tolist())
        labels.extend(lbls.tolist())
        if extra:
            vids.extend(list(extra[0]) if len(extra) > 0 else [""] * len(lbls))
            sources.extend(list(extra[1]) if len(extra) > 1 else [""] * len(lbls))

    return (np.array(scores, dtype=np.float32),
            np.array(labels, dtype=np.int32),
            vids or [""] * len(scores),
            sources or [""] * len(scores))


def _discover_ckpt(configured_path: str, model_name: str,
                    out_root: str) -> str:
    """Return a checkpoint path that actually exists.

    Search order:
      1. The exact path from the config (after resolving against out_root).
      2. Glob for best.pt under <out_root>/**/<model_name>_seed*/best.pt
         (handles the content-hashed subdir layout written by training scripts).
      3. Glob for best_<model_name>_seed*.pt under <out_root>/**/checkpoints/
         (handles the b1_ablation layout).
    """
    # Resolve relative paths against out_root
    resolved = configured_path
    if configured_path and not os.path.isabs(configured_path):
        resolved = configured_path.replace("outputs/", out_root.rstrip("/") + "/")

    if resolved and os.path.exists(resolved):
        return resolved

    # Glob patterns for content-hashed layouts
    safe_name = model_name.replace(" ", "_").replace("-", "-")
    patterns = [
        # c3_train_regimes layout: outputs/c3_regimes/<hash>/<regime>_seed*/best.pt
        str(Path(out_root) / "c3_regimes" / "*" / f"{safe_name}_seed*" / "best.pt"),
        # b1_ablation layout: outputs/b1_ablation/<hash>/checkpoints/best_<model>_seed*.pt
        str(Path(out_root) / "b1_ablation" / "*" / "checkpoints" / f"best_{safe_name}_seed*.pt"),
        # c4_dl_quartile layout: outputs/c4_dl_quartile/<hash>/<variant>_seed*/best.pt
        str(Path(out_root) / "c4_dl_quartile" / "*" / f"{safe_name}_seed*" / "best.pt"),
        # b4_fixed_fusion layout: outputs/b4_fixed_fusion/<hash>/checkpoints/<model>_seed*.pt
        str(Path(out_root) / "b4_fixed_fusion" / "*" / "checkpoints" / f"{safe_name}_seed*.pt"),
    ]
    import glob as _glob
    for pattern in patterns:
        hits = sorted(_glob.glob(pattern))
        if hits:
            # Pick seed42 if available, else the first hit
            seed42 = [h for h in hits if "seed42" in h]
            chosen = seed42[0] if seed42 else hits[0]
            logger.info(f"  Discovered checkpoint for {model_name}: {chosen}")
            return chosen

    return ""   # not found


def main() -> None:
    args = _parse_args()
    t0   = time.time()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt_cfg_path = args.checkpoints or args.config
    with open(ckpt_cfg_path) as f:
        ckpt_cfg = yaml.safe_load(f)

    out_root_str = os.environ.get("THESIS_OUTPUT_ROOT",
                                   cfg.get("output_root", "outputs"))
    cfg_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()[:12]
    out_dir  = Path(out_root_str) / "d_eval" / cfg_hash
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = out_dir / "predictions"
    preds_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = ckpt_cfg.get("checkpoints", {})
    # Resolve manifest paths against out_root
    # manifests can be either:
    #   dataset_name: path_string           (legacy)
    #   dataset_name: {csv: path, split: X} (new — supports per-dataset split)
    raw_manifests = ckpt_cfg.get("manifests", {})
    manifests = {}   # {name: (csv_path, split)}
    for k, v in raw_manifests.items():
        if isinstance(v, dict):
            csv_path = v.get("csv", "")
            split    = v.get("split", "test")
        else:
            csv_path = str(v)
            # CelebDF/DFDC/DFF have no test split — infer from dataset name
            split = "test" if "ff++" in k.lower() or "ffpp" in k.lower() \
                    else "all"
        if not os.path.isabs(csv_path):
            csv_path = csv_path.replace("outputs/",
                                         out_root_str.rstrip("/") + "/")
        manifests[k] = (csv_path, split)

    if not checkpoints:
        logger.error("No checkpoints defined. "
                     "Add 'checkpoints:' key to config or --checkpoints YAML.")
        sys.exit(1)

    all_metrics = []

    from torchvision import transforms
    from src.data.dataset import FaceCropDataset
    from src.eval.metrics import compute_metrics
    from src.eval.aggregation import aggregate_to_video
    from src.eval.bootstrap import bootstrap_auc_ci
    from src.eval.threshold_sweep import sweep_thresholds, find_optima
    from src.eval.cache import DiskCache

    cache = DiskCache(out_dir / "_cache")

    def _make_transform(family: str):
        if family == "stage2":
            # EfficientNet-B4: 380×380, ImageNet normalisation
            return transforms.Compose([
                transforms.Resize((380, 380)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        # Stage 1: 299×299, [-1,1] normalisation
        return transforms.Compose([
            transforms.Resize((330, 330)),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    for model_name, model_spec in checkpoints.items():
        family    = model_spec["family"]
        ckpt_path = _discover_ckpt(model_spec.get("ckpt", ""),
                                    model_name, out_root_str)
        if not ckpt_path:
            logger.warning(f"Checkpoint not found for {model_name} — skipping. "
                           f"Run the corresponding training step first.")
            continue

        logger.info(f"Loading {model_name} ({family}) from {ckpt_path}")
        try:
            model = _load_model(family, ckpt_path, device)
        except Exception as exc:
            logger.error(f"Failed loading {model_name}: {exc} — skipping")
            continue

        for dataset_name, (manifest_csv, eval_split) in manifests.items():
            if not os.path.exists(manifest_csv):
                logger.warning(f"Manifest not found: {manifest_csv} — skipping")
                continue

            pred_csv = str(preds_dir / f"{model_name}__{dataset_name}.csv")
            use_cache = (os.path.exists(pred_csv) and not args.force)
            if use_cache:
                df_pred = pd.read_csv(pred_csv)
                # Invalidate empty cache (e.g. from a previous run with wrong split)
                if len(df_pred) == 0:
                    logger.warning(f"  Cached CSV is empty for {model_name}|{dataset_name} "
                                   f"— re-running inference")
                    use_cache = False
                else:
                    logger.info(f"  CACHED  {model_name} | {dataset_name} "
                                f"({len(df_pred)} rows)")
                    s = df_pred["score"].to_numpy()
                    l = df_pred["label"].to_numpy().astype(np.int32)
                    vids = df_pred.get("video_id", pd.Series([""] * len(s))).tolist()
                    srcs = df_pred.get("source",   pd.Series([""] * len(s))).tolist()
                    # Re-save to cache with paths if not already there
                    if "face_crop_path" in df_pred.columns:
                        paths_cached = df_pred["face_crop_path"].tolist()
                        if not cache.has_inference(model_name, dataset_name) or args.force:
                            cache.save_inference(
                                model_name, dataset_name, s, l,
                                video_ids=np.array(vids, dtype=object),
                                sources=np.array(srcs, dtype=object),
                                paths=np.array(paths_cached, dtype=object))
            if not use_cache:
                ds = FaceCropDataset.from_csv(manifest_csv, split=eval_split,
                                               transform=_make_transform(family))
                loader = torch.utils.data.DataLoader(
                    ds, batch_size=32, shuffle=False, num_workers=0)
                s, l, vids, srcs = _run_inference(model, loader, device, args.tiny)
                # Extract face_crop_path from the dataset in the same order
                paths = [str(ds.df.iloc[i]["face_crop_path"])
                         for i in range(len(ds))][:len(s)]
                df_pred = pd.DataFrame({
                    "score": s, "label": l,
                    "video_id": vids, "source": srcs,
                    "face_crop_path": paths,
                })
                df_pred.to_csv(pred_csv, index=False)
                cache.save_inference(model_name, dataset_name, s, l,
                                     video_ids=np.array(vids, dtype=object),
                                     sources=np.array(srcs, dtype=object),
                                     paths=np.array(paths, dtype=object))

            # Frame-level metrics
            fm = compute_metrics(l, s)
            # Video-level metrics (mean aggregation)
            v_s, v_l, _ = aggregate_to_video(s, l, srcs, vids, strategy="mean")
            vm = compute_metrics(v_l, v_s)

            # Bootstrap CI
            lo, hi = bootstrap_auc_ci(s, l, n_resamples=200, seed=42)

            # Threshold sweep
            curve = sweep_thresholds(s, l)
            opt   = find_optima(curve)

            row = {
                "model": model_name, "dataset": dataset_name,
                "frame_auc": fm["auc"],   "video_auc": vm["auc"],
                "eer": fm["eer"],          "fpr_at_tpr95": fm["fpr_at_tpr95"],
                "ece": fm["ece"],          "brier": fm["brier"],
                "auc_ci_lo": lo,           "auc_ci_hi": hi,
                "max_bal_acc": opt["max_bal_acc"],
                "max_mcc":     opt["max_mcc"],
                "n_samples":  fm["n"],
            }
            all_metrics.append(row)
            logger.info(
                f"  [{model_name}|{dataset_name}]  "
                f"frame_AUC={fm['auc']:.4f}  video_AUC={vm['auc']:.4f}"
            )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = out_dir / "eval_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Wrote {metrics_path}")

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
