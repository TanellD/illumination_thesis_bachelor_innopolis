"""
src/train/stage2_loop.py
========================
Stage 2 training loop for EfficientNet-B4 + T-SBI regimes.

Design choices (locked per thesis §5):
  - BCEWithLogitsLoss (single logit output)
  - AdamW, lr=1e-4, weight_decay=1e-4
  - Linear warmup (1 epoch) → cosine decay
  - Class-balanced WeightedRandomSampler (mandatory — do not skip)
  - AMP mixed precision on CUDA
  - Checkpoint saved on best val *video-level* AUC
  - Grouping key for video-level AUC: (source, video_id) — NOT just video_id

Ported from scripts/T_SBI/train.py (the uncommented second version).
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
INPUT_SIZE    = 380


# ── albumentations transform ──────────────────────────────────────────────────

def _compat_image_compression(qmin: int, qmax: int, p: float):
    import albumentations as A
    try:
        return A.ImageCompression(quality_range=(qmin, qmax), p=p)
    except TypeError:
        return A.ImageCompression(quality_lower=qmin, quality_upper=qmax, p=p)


def _compat_gauss_noise(p: float):
    import albumentations as A
    try:
        return A.GaussNoise(std_range=(5.0 / 255.0, 30.0 / 255.0), p=p)
    except TypeError:
        return A.GaussNoise(var_limit=(5.0, 30.0), p=p)


def build_transform(train: bool):
    """Return an albumentations Compose (or a plain torchvision fallback)."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        if train:
            return A.Compose([
                A.LongestMaxSize(max_size=int(INPUT_SIZE * 1.15)),
                A.PadIfNeeded(INPUT_SIZE, INPUT_SIZE, border_mode=0),
                A.RandomCrop(INPUT_SIZE, INPUT_SIZE),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(0.2, 0.2, p=1.0),
                    A.HueSaturationValue(10, 20, 10, p=1.0),
                ], p=0.5),
                # Class-invariant heavy degradations — mirror jitter_source_only
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.Downscale(scale_min=0.5, scale_max=0.9,
                                interpolation=0, p=1.0),
                ], p=0.5),
                _compat_image_compression(60, 95, p=0.5),
                _compat_gauss_noise(p=0.3),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.LongestMaxSize(max_size=INPUT_SIZE),
                A.PadIfNeeded(INPUT_SIZE, INPUT_SIZE, border_mode=0),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
    except ImportError:
        return None   # caller must handle fallback


# ── dataset ───────────────────────────────────────────────────────────────────

class Stage2FaceCropDataset(Dataset):
    """Reads a manifest CSV (DictReader rows) and returns
    (image_tensor, label, video_id, source) for Stage 2 training.
    """

    def __init__(self, rows: List[dict], train: bool = True):
        self.rows = rows
        self.tf = build_transform(train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        import cv2
        r = self.rows[idx]
        img = cv2.imread(r["face_crop_path"])
        if img is None:
            img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.tf is not None:
            img = self.tf(image=img)["image"]
        else:
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            mean = np.array(IMAGENET_MEAN, dtype=np.float32)
            std  = np.array(IMAGENET_STD,  dtype=np.float32)
            img = (img.astype(np.float32) / 255.0 - mean) / std
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()

        label = int(r["label"])
        vid   = r.get("video_id", "")
        src   = r.get("source",   "")
        return img, label, vid, src


def balanced_sampler(rows: List[dict]) -> WeightedRandomSampler:
    """Class-balanced WeightedRandomSampler (mandatory — see CLAUDE.md §C.3)."""
    n0 = sum(1 for r in rows if int(r["label"]) == 0)
    n1 = sum(1 for r in rows if int(r["label"]) == 1)
    w0 = 1.0 / max(1, n0)
    w1 = 1.0 / max(1, n1)
    weights = [w0 if int(r["label"]) == 0 else w1 for r in rows]
    return WeightedRandomSampler(weights, num_samples=len(weights),
                                  replacement=True)


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_warmup_lr(
    step: int, total_steps: int, warmup_steps: int, base_lr: float
) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * p))


# ── eval pass ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_stage2(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Return frame_auc and video_auc (grouped by (source, video_id))."""
    model.eval()
    all_scores: list = []
    all_labels: list = []
    all_vids:   list = []
    all_srcs:   list = []

    for img, y, vid, src in loader:
        img = img.to(device, non_blocking=True)
        prob = torch.sigmoid(model(img).squeeze(1)).float().cpu().numpy()
        all_scores.append(prob)
        all_labels.append(y.numpy())
        all_vids.extend(vid)
        all_srcs.extend(src)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    try:
        frame_auc = float(roc_auc_score(labels, scores))
    except ValueError:
        frame_auc = float("nan")

    by_key: dict = defaultdict(list)
    by_key_label: dict = {}
    for s, l, v, src in zip(scores, labels, all_vids, all_srcs):
        key = (src, v)
        by_key[key].append(s)
        by_key_label[key] = l

    v_scores = np.array([np.mean(by_key[k]) for k in by_key])
    v_labels = np.array([by_key_label[k] for k in by_key])
    try:
        video_auc = float(roc_auc_score(v_labels, v_scores))
    except ValueError:
        video_auc = float("nan")

    return {"frame_auc": frame_auc, "video_auc": video_auc}


# ── training loop ─────────────────────────────────────────────────────────────

def train_stage2(
    model: nn.Module,
    train_rows: List[dict],
    val_rows:   List[dict],
    device: torch.device,
    out_dir: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    seed: int = 42,
) -> dict:
    """Full Stage 2 training run.

    Saves best.pt (by val video AUC) and last.pt to out_dir.
    Returns history dict with per-epoch metrics.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    train_ds = Stage2FaceCropDataset(train_rows, train=True)
    val_ds   = Stage2FaceCropDataset(val_rows,   train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=balanced_sampler(train_rows),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, len(train_loader))

    best_auc = -1.0
    step = 0
    history: list = []

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n_seen = 0

        for img, y, *_ in train_loader:
            img = img.to(device, non_blocking=True)
            y   = y.float().to(device, non_blocking=True)

            lr_now = cosine_warmup_lr(step, total_steps, warmup_steps, lr)
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logit = model(img).squeeze(1)
                loss  = loss_fn(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item() * img.size(0)
            n_seen  += img.size(0)
            step    += 1

        val_metrics = evaluate_stage2(model, val_loader, device)
        tr_loss = running / max(1, n_seen)
        epoch_record = {
            "epoch":        ep,
            "train_loss":   tr_loss,
            "val_frame_auc": val_metrics["frame_auc"],
            "val_video_auc": val_metrics["video_auc"],
            "lr":           lr_now,
            "time_s":       round(time.time() - t0, 1),
        }
        history.append(epoch_record)
        logger.info(
            f"ep {ep:3d} | loss {tr_loss:.4f} | "
            f"val_frame_auc {val_metrics['frame_auc']:.4f} | "
            f"val_video_auc {val_metrics['video_auc']:.4f} | "
            f"lr {lr_now:.2e}"
        )

        current = val_metrics["video_auc"]
        if not math.isnan(current) and current > best_auc:
            best_auc = current
            torch.save(
                {"model": model.state_dict(),
                 "epoch": ep,
                 "val_video_auc": current,
                 "val_frame_auc": val_metrics["frame_auc"]},
                os.path.join(out_dir, "best.pt"),
            )
            logger.info(f"  -> new best ({current:.4f}), saved best.pt")

    # Guarantee best.pt exists
    best_path = os.path.join(out_dir, "best.pt")
    if not os.path.exists(best_path):
        torch.save(
            {"model": model.state_dict(),
             "epoch": epochs - 1,
             "val_video_auc": float("nan"),
             "note": "fallback: no valid ranking metric"},
            best_path,
        )
    torch.save(
        {"model": model.state_dict(), "epoch": epochs - 1},
        os.path.join(out_dir, "last.pt"),
    )

    return {"history": history, "best_val_video_auc": best_auc}
