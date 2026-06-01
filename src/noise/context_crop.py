"""
src/noise/context_crop.py
=========================
Context-crop experiment (B.3).

Applies the bottleneck diagnostic to two bbox expansion strategies side-by-side:
  - Tight crop  (1.3× expansion)  — original pipeline default
  - Context crop (2.7× expansion) — captures blending boundary

For each strategy, extracts Noiseprint++ noise maps and runs 4 classifiers
(L1 LogReg mean_abs, L2 LogReg 6-feat, L3 TinyConv raw, L4 TinyConv+InstanceNorm)
via 5-fold cross-validation.  Also computes a corpus-wise SNR:
    SNR = |mean_real - mean_fake| / ((std_real + std_fake) / 2)

Both the tight and context SNRs are reported so the analyst can assess
whether broader context actually helps the noise signal.

Critical: noise extraction must run on the FULL frame, then crop.
This module enforces that order by accepting full-frame paths separately
from the bbox coordinates.  Do not feed already-cropped images.

Ported from scripts/stage1_noise_channel/context_crop_size.py.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

N_FOLDS = 5


# ── crop strategies ───────────────────────────────────────────────────────────

def crop_tight(
    frame: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    target_size: int = 299,
    expansion: float = 1.3,
) -> np.ndarray:
    """1.3× face crop (original pipeline default)."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side   = max(x2 - x1, y2 - y1) * expansion / 2.0
    nx1 = max(0, int(cx - side));  ny1 = max(0, int(cy - side))
    nx2 = min(W, int(cx + side));  ny2 = min(H, int(cy + side))
    crop = frame[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (target_size, target_size),
                      interpolation=cv2.INTER_LINEAR)


def crop_context(
    frame: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    target_size: int = 299,
    expansion: float = 2.7,
) -> np.ndarray:
    """2.7× context crop with BORDER_REFLECT_101 padding."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side   = max(x2 - x1, y2 - y1) * expansion / 2.0
    nx1, ny1 = int(cx - side), int(cy - side)
    nx2, ny2 = int(cx + side), int(cy + side)
    pl = max(0, -nx1);  pt = max(0, -ny1)
    pr = max(0, nx2 - W); pb = max(0, ny2 - H)
    if any([pl, pt, pr, pb]):
        frame = cv2.copyMakeBorder(frame, pt, pb, pl, pr,
                                   cv2.BORDER_REFLECT_101)
        nx1 += pl; ny1 += pt; nx2 += pl; ny2 += pt
    crop = frame[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return cv2.resize(frame, (target_size, target_size),
                          interpolation=cv2.INTER_LINEAR)
    return cv2.resize(crop, (target_size, target_size),
                      interpolation=cv2.INTER_LINEAR)


# ── feature extraction ────────────────────────────────────────────────────────

def scalar_features(noise_t: torch.Tensor) -> np.ndarray:
    n = noise_t.numpy().astype(np.float64).flatten()
    abs_n = np.abs(n)
    mu    = n.mean()
    sigma = n.std() + 1e-12
    return np.array([
        abs_n.mean(),
        sigma,
        (n ** 2).mean(),
        abs_n.max(),
        ((n - mu) ** 4).mean() / (sigma ** 4) - 3.0,
        noise_t.shape[-1] * noise_t.shape[-2],
    ], dtype=np.float64)


def snr(noise_tensors: List[torch.Tensor], labels: List[int]) -> float:
    real_m = [noise_tensors[i].abs().mean().item()
               for i, l in enumerate(labels) if l == 1]
    fake_m = [noise_tensors[i].abs().mean().item()
               for i, l in enumerate(labels) if l == 0]
    if not real_m or not fake_m:
        return float("nan")
    rm, fm = float(np.mean(real_m)), float(np.mean(fake_m))
    rs, fs = float(np.std(real_m)),  float(np.std(fake_m))
    return abs(rm - fm) / ((rs + fs) / 2.0 + 1e-9)


# ── TinyConvNet (same 2-layer net as bottleneck diagnostic) ───────────────────

class _TinyConv(torch.nn.Module):
    def __init__(self, use_inorm: bool = False):
        super().__init__()
        self.use_in = use_inorm
        if use_inorm:
            self.inorm = torch.nn.InstanceNorm2d(1, affine=True)
        self.conv1 = torch.nn.Conv2d(1, 16, 5, stride=4, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.pool  = torch.nn.AdaptiveAvgPool2d(1)
        self.fc    = torch.nn.Linear(32, 2)

    def forward(self, x):
        if self.use_in:
            x = self.inorm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.fc(self.pool(x).flatten(1))


def _train_tiny_conv(
    noise_tensors: List[torch.Tensor],
    labels: List[int],
    skf: StratifiedKFold,
    use_inorm: bool,
    epochs: int = 12,
) -> List[float]:
    y = np.array(labels)
    aucs = []
    for tr_idx, vl_idx in skf.split(np.zeros(len(y)), y):
        tr_n = F.interpolate(
            torch.stack([noise_tensors[i] for i in tr_idx]),
            size=128, mode="bilinear", align_corners=False)
        vl_n = F.interpolate(
            torch.stack([noise_tensors[i] for i in vl_idx]),
            size=128, mode="bilinear", align_corners=False)
        tr_y = torch.tensor(y[tr_idx], dtype=torch.long)
        vl_y = torch.tensor(y[vl_idx], dtype=torch.long)

        model = _TinyConv(use_inorm=use_inorm)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit  = torch.nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(tr_n, tr_y),
                             batch_size=32, shuffle=True)
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(vl_n), 1)[:, 1].numpy()
        aucs.append(float(roc_auc_score(y[vl_idx], probs)))
    return aucs


# ── main diagnostic ───────────────────────────────────────────────────────────

def run_context_crop_diagnostic(
    manifest_csv: str,
    output_dir: str,
    noise_weights_path: Optional[str],
    n_samples: int = 2000,
    split: Optional[str] = "test",
    tight_expansion: float = 1.3,
    context_expansion: float = 2.7,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the context-crop diagnostic for experiment B.3.

    For 2000 FF++ samples (matching thesis), extracts noise under two crop
    strategies and runs 4 classifier levels via 5-fold CV.

    Returns a tidy DataFrame with columns:
        crop_strategy, level, mean_auc, std_auc, snr
    """
    import os, random
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    df = pd.read_csv(manifest_csv)
    if split and "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    # Load noise model
    from src.noise.precompute import load_noise_model
    device = torch.device("cpu")
    noise_model = load_noise_model(noise_weights_path, device)

    # Determine full-frame column
    ff_col = next((c for c in ("full_frame_path", "filepath") if c in df.columns),
                  None)
    face_col = "face_crop_path" if "face_crop_path" in df.columns else "image_path"

    # Extract noise under both strategies for each sample
    tight_tensors,   context_tensors = [], []
    labels_out = []

    for _, row in df.iterrows():
        label = int(row["label"])
        # Use full frame if available; else fall back to face crop
        img_path = str(row[ff_col]) if ff_col else str(row[face_col])
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract noise on full frame (extract-then-crop order, KNOWN_QUIRKS #3)
        t = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
        with torch.no_grad():
            noise_full = noise_model(t.unsqueeze(0)).squeeze(0)   # [1, H, W]

        # Crop noise map using both strategies
        bbox = (float(row["x1"]), float(row["y1"]),
                float(row["x2"]), float(row["y2"]))

        # Crop the noise map directly (same bbox as RGB crop)
        H, W = frame.shape[:2]
        nh, nw = noise_full.shape[-2], noise_full.shape[-1]

        def _crop_noise(expansion: float) -> torch.Tensor:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            side = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * expansion / 2.0
            nx1 = max(0, int(cx - side)); ny1 = max(0, int(cy - side))
            nx2 = min(nw, int(cx + side)); ny2 = min(nh, int(cy + side))
            crop = noise_full[:, ny1:ny2, nx1:nx2]
            if crop.numel() == 0:
                crop = noise_full
            return F.interpolate(crop.unsqueeze(0), size=299,
                                  mode="bilinear", align_corners=False).squeeze(0)

        tight_tensors.append(_crop_noise(tight_expansion))
        context_tensors.append(_crop_noise(context_expansion))
        labels_out.append(label)

    logger.info(f"Extracted {len(labels_out)} noise samples "
                f"({sum(l==1 for l in labels_out)} real, "
                f"{sum(l==0 for l in labels_out)} fake)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    results = []

    for strategy, tensors in [("tight", tight_tensors),
                                ("context", context_tensors)]:
        sig = snr(tensors, labels_out)
        y   = np.array(labels_out)
        X   = np.stack([scalar_features(t) for t in tensors])

        # L1: LogReg on mean_abs
        aucs = []
        for tr, vl in skf.split(X, y):
            sc = StandardScaler().fit(X[tr, 0:1])
            clf = LogisticRegression(max_iter=1000).fit(
                sc.transform(X[tr, 0:1]), y[tr])
            aucs.append(roc_auc_score(y[vl],
                clf.predict_proba(sc.transform(X[vl, 0:1]))[:, 1]))
        results.append({"crop_strategy": strategy, "level": "L1_mean_abs",
                         "mean_auc": float(np.mean(aucs)),
                         "std_auc":  float(np.std(aucs)),  "snr": sig})

        # L2: LogReg on 6 features
        aucs = []
        for tr, vl in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000).fit(
                sc.transform(X[tr]), y[tr])
            aucs.append(roc_auc_score(y[vl],
                clf.predict_proba(sc.transform(X[vl]))[:, 1]))
        results.append({"crop_strategy": strategy, "level": "L2_6feat",
                         "mean_auc": float(np.mean(aucs)),
                         "std_auc":  float(np.std(aucs)), "snr": sig})

        # L3: TinyConv raw
        aucs = _train_tiny_conv(tensors, labels_out, skf, use_inorm=False)
        results.append({"crop_strategy": strategy, "level": "L3_conv_raw",
                         "mean_auc": float(np.mean(aucs)),
                         "std_auc":  float(np.std(aucs)), "snr": sig})

        # L4: TinyConv + InstanceNorm
        aucs = _train_tiny_conv(tensors, labels_out, skf, use_inorm=True)
        results.append({"crop_strategy": strategy, "level": "L4_conv_inorm",
                         "mean_auc": float(np.mean(aucs)),
                         "std_auc":  float(np.std(aucs)), "snr": sig})

        logger.info(f"  {strategy}: SNR={sig:.4f}  "
                    f"L1={results[-4]['mean_auc']:.4f}  "
                    f"L2={results[-3]['mean_auc']:.4f}  "
                    f"L3={results[-2]['mean_auc']:.4f}  "
                    f"L4={results[-1]['mean_auc']:.4f}")

    out_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "context_crop_results.csv")
    out_df.to_csv(csv_path, index=False)
    logger.info(f"Wrote {csv_path}")
    return out_df
