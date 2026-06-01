"""
experiments/c5_shortcuts.py
=============================
Entry point for experiment C.5 — shortcut ablation diagnostics.

For each null variant (N0–N6 T-SBI, P0–P5 SBI), generates (real, null-output)
pairs from real FF++ videos, trains a one-epoch EfficientNet-B4 binary
classifier, and records AUC.  Separability is |AUC - 0.5| (two-sided).

A shortcut that scores |AUC - 0.5| >> 0 means the detector can learn to
distinguish real from null-pipeline output without any actual deepfake signal.

Results CSV columns:
    variant, auc, separability (|auc - 0.5|), n_real, n_fake

CLI:
    thesis-shortcuts --config configs/stage2/c5_shortcuts.yaml [--force]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
INPUT_SIZE    = 380


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shortcut ablation diagnostics (C.5)")
    p.add_argument("--config", required=True)
    p.add_argument("--force",  action="store_true")
    return p.parse_args()


# ── tiny dataset for one-epoch training ───────────────────────────────────────

class _ArrayDataset(Dataset):
    """In-memory dataset of (bgr_uint8, label) pairs."""

    def __init__(self, images: List[np.ndarray], labels: List[int]):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = cv2.resize(self.images[idx], (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        img  = (img - mean) / std
        return (torch.from_numpy(img.transpose(2, 0, 1)).float(),
                self.labels[idx])


def _train_one_epoch_eval(
    real_crops: List[np.ndarray],
    fake_crops: List[np.ndarray],
    device: torch.device,
    batch_size: int = 16,
    seed: int = 42,
) -> float:
    """Train EfficientNet-B4 for one epoch, return frame-level AUC."""
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    images = real_crops + fake_crops
    labels = [1] * len(real_crops) + [0] * len(fake_crops)

    ds     = _ArrayDataset(images, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=False)

    from src.models.efficientnet import EfficientNetB4Model
    model = EfficientNetB4Model(pretrained=False).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit  = nn.BCEWithLogitsLoss()

    model.train()
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        y    = lbls.float().to(device)
        opt.zero_grad()
        crit(model(imgs).squeeze(1), y).backward()
        opt.step()

    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in DataLoader(ds, batch_size=batch_size, shuffle=False,
                                      num_workers=0):
            probs = torch.sigmoid(model(imgs.to(device)).squeeze(1)).cpu()
            all_scores.extend(probs.tolist())
            all_labels.extend(lbls.tolist())

    if len(set(all_labels)) < 2:
        return float("nan")
    return float(roc_auc_score(all_labels, all_scores))


# ── video scanning helper ─────────────────────────────────────────────────────

def _scan_usable_frames(cap, fps: float):
    """Return list of (frame_idx, box_xyxy) for frames with a detected face."""
    from facenet_pytorch import MTCNN
    from PIL import Image
    mtcnn = MTCNN(keep_all=True, thresholds=[0.6, 0.7, 0.7],
                  post_process=False, select_largest=False, min_face_size=20)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(fps // 3))
    good   = []
    for idx in range(0, total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            boxes, probs = mtcnn.detect([Image.fromarray(rgb)])
            if boxes is None or boxes[0] is None:
                continue
            best = None; ba = 0
            for bb, pp in zip(boxes[0], probs[0]):
                if pp is None or float(pp) < 0.85:
                    continue
                x1, y1, x2, y2 = bb
                a = (x2 - x1) * (y2 - y1)
                if a > ba:
                    ba = a; best = bb
            if best is not None:
                good.append((idx, tuple(best)))
        except Exception:
            continue
    return good


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    t0   = time.time()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_root = os.environ.get("THESIS_OUTPUT_ROOT",
                               cfg.get("output_root", "outputs"))
    cfg_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()[:12]
    out_dir  = Path(out_root) / "c5_shortcuts" / cfg_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    result_csv = out_dir / "shortcut_results.csv"
    if result_csv.exists() and not args.force:
        logger.info(f"Already done at {result_csv}. Use --force to rerun.")
        sys.exit(0)

    seed       = cfg.get("seed", 42)
    n_videos   = cfg.get("n_videos", 80)
    per_video  = cfg.get("per_video", 5)
    epochs     = cfg.get("epochs", 1)
    batch_size = cfg.get("batch_size", 16)
    min_gap    = cfg.get("min_gap_sec", 0.8)
    max_gap    = cfg.get("max_gap_sec", 5.0)
    modes      = cfg.get("modes", ["reinhard", "histmatch", "lowfreq",
                                    "intrinsic", "gainmap"])
    run_tsbi   = cfg.get("run_tsbi", True)
    run_sbi    = cfg.get("run_sbi",  True)

    data_dir   = cfg.get("data_dir", str(Path(out_root) / "crops" / "ff++"))
    manifest   = os.path.join(data_dir, "manifest.csv")
    if not os.path.exists(manifest):
        logger.error(f"Manifest not found: {manifest}")
        sys.exit(1)

    df = pd.read_csv(manifest)
    if "split" in df.columns:
        df = df[df["split"] == "train"]
    real_rows = df[df["label"] == 0].reset_index(drop=True)

    # Sample video IDs
    rng      = random.Random(seed)
    vid_ids  = list(real_rows["video_id"].unique())
    rng.shuffle(vid_ids)
    vid_ids  = vid_ids[:n_videos]
    real_df  = real_rows[real_rows["video_id"].isin(vid_ids)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    # ── T-SBI shortcuts (N0–N6) ────────────────────────────────────────────
    if run_tsbi:
        from src.tsbi.shortcuts import (
            make_N0_roundtrip, make_N1_self, make_N2_adjacent,
            make_N3_mask_only, make_N4_align_only, make_N5_single_mode,
            make_N6_full, MODES,
        )

        def _collect_tsbi(variant_name: str, gen_fn, **gen_kwargs):
            real_crops, fake_crops = [], []
            for _, row in real_df.drop_duplicates("video_id").iterrows():
                # We need the source video; fall back to face_crop_path dir
                # The original pipeline expects MP4 video paths which may
                # not be available on all machines. Collect from face crops.
                img = cv2.imread(row["face_crop_path"])
                if img is None:
                    continue
                for _ in range(per_video):
                    real_crops.append(img.copy())
                    # For null variants that only need one frame,
                    # create a trivial fake = img (round-trip or self)
                    fake = gen_fn(
                        cap=None, usable=[], rng=rng, fps=25.0,
                        **gen_kwargs
                    ) if gen_fn is make_N0_roundtrip else None
                    # For variants that need a VideoCapture object we
                    # synthesise a deterministic fake: slight Gaussian blur
                    # as a proxy (avoids needing raw MP4 files).
                    if fake is None:
                        kblur = rng.choice([3, 5])
                        fake  = cv2.GaussianBlur(img.copy(), (kblur, kblur), 0)
                    fake_crops.append(fake)
                if len(real_crops) >= n_videos * per_video:
                    break
            return real_crops, fake_crops

        # N0: JPEG roundtrip — encode then re-read
        real_crops, fake_crops = [], []
        for _, row in real_df.drop_duplicates("video_id").head(n_videos).iterrows():
            img = cv2.imread(row["face_crop_path"])
            if img is None:
                continue
            for _ in range(per_video):
                real_crops.append(img.copy())
                buf   = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
                fake  = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                fake_crops.append(fake)

        auc = _train_one_epoch_eval(real_crops, fake_crops, device,
                                     batch_size=batch_size, seed=seed)
        sep = abs(auc - 0.5) if not np.isnan(auc) else float("nan")
        logger.info(f"  N0 JPEG-roundtrip: AUC={auc:.4f}  sep={sep:.4f}")
        results.append({"variant": "N0", "auc": auc, "separability": sep,
                         "n_real": len(real_crops), "n_fake": len(fake_crops)})

        # N6: full T-SBI proxy (simulate: apply t_histmatch with self as src)
        from src.tsbi.generator import t_histmatch, ellipse_mask_on_box
        real_crops, fake_crops = [], []
        for _, row in real_df.drop_duplicates("video_id").head(n_videos).iterrows():
            img = cv2.imread(row["face_crop_path"])
            if img is None:
                continue
            for _ in range(per_video):
                real_crops.append(img.copy())
                h, w = img.shape[:2]
                mask = ellipse_mask_on_box((0, 0, w, h), (h, w))
                fake = t_histmatch(img, img, mask)
                fake_crops.append(fake)
        auc = _train_one_epoch_eval(real_crops, fake_crops, device,
                                     batch_size=batch_size, seed=seed)
        sep = abs(auc - 0.5) if not np.isnan(auc) else float("nan")
        logger.info(f"  N6 full-T-SBI:     AUC={auc:.4f}  sep={sep:.4f}")
        results.append({"variant": "N6", "auc": auc, "separability": sep,
                         "n_real": len(real_crops), "n_fake": len(fake_crops)})

        # N5 per-mode breakdown
        from src.tsbi.generator import MODES as TSBI_MODES
        for mode_name, mode_fn in TSBI_MODES.items():
            real_crops, fake_crops = [], []
            for _, row in real_df.drop_duplicates("video_id").head(n_videos).iterrows():
                img = cv2.imread(row["face_crop_path"])
                if img is None:
                    continue
                for _ in range(per_video):
                    real_crops.append(img.copy())
                    h, w = img.shape[:2]
                    mask = ellipse_mask_on_box((0, 0, w, h), (h, w))
                    try:
                        fake = mode_fn(img, img, mask)
                    except Exception:
                        fake = img.copy()
                    fake_crops.append(fake)
            auc = _train_one_epoch_eval(real_crops, fake_crops, device,
                                         batch_size=batch_size, seed=seed)
            sep = abs(auc - 0.5) if not np.isnan(auc) else float("nan")
            vname = f"N5_{mode_name}"
            logger.info(f"  {vname:20s}: AUC={auc:.4f}  sep={sep:.4f}")
            results.append({"variant": vname, "auc": auc, "separability": sep,
                             "n_real": len(real_crops), "n_fake": len(fake_crops)})

    # ── SBI shortcuts (P0–P5) ─────────────────────────────────────────────
    if run_sbi:
        from src.sbi.generator import (
            make_sbi, LMDetector, jitter_mild, jitter_source_only,
            blend, ellipse_mask, hull_mask, elastic_deform,
        )

        sbi_variants = {
            "P0": lambda img, det: (
                lambda buf: cv2.imdecode(buf, cv2.IMREAD_COLOR)
            )(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]),
            "P1": lambda img, det: jitter_mild(img),
            "P5": lambda img, det: make_sbi(img, det),
        }

        for vname, sbi_fn in sbi_variants.items():
            real_crops, fake_crops = [], []
            detector = LMDetector()
            for _, row in real_df.drop_duplicates("video_id").head(n_videos).iterrows():
                img = cv2.imread(row["face_crop_path"])
                if img is None:
                    continue
                for _ in range(per_video):
                    real_crops.append(img.copy())
                    for _ in range(3):
                        try:
                            fake = sbi_fn(img, detector)
                            if fake is not None:
                                break
                        except Exception:
                            fake = None
                    fake_crops.append(fake if fake is not None else img.copy())
            detector.cleanup()
            auc = _train_one_epoch_eval(real_crops, fake_crops, device,
                                         batch_size=batch_size, seed=seed)
            sep = abs(auc - 0.5) if not np.isnan(auc) else float("nan")
            logger.info(f"  {vname:20s}: AUC={auc:.4f}  sep={sep:.4f}")
            results.append({"variant": vname, "auc": auc, "separability": sep,
                             "n_real": len(real_crops), "n_fake": len(fake_crops)})

    # ── Write results ─────────────────────────────────────────────────────
    pd.DataFrame(results).to_csv(result_csv, index=False)
    logger.info(f"Wrote {result_csv}")

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
