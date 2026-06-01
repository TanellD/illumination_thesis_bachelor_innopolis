"""
src/noise/bottleneck.py
=======================
Seven-level bottleneck diagnostic for noise-residual analysis (experiment B.2).

Tests progressively complex classifiers on precomputed noise crops to
localise where the noise signal is destroyed:

  L1: logistic regression on mean_abs alone (1 feature)
  L2: logistic regression on 6 scalar features
  L3: MLP (6→32→16→2) on those 6 features
  L4: logistic regression on AdaptiveAvgPool(InstanceNorm(noise))
  L5: logistic regression on AdaptiveAvgPool(raw noise) — no InstanceNorm
  L6: 2-layer ConvNet on raw noise (resized to 128×128)
  L7: 2-layer ConvNet on InstanceNorm(noise)

All levels use 5-fold stratified CV on ≤3000 test samples.
Do NOT change the level definitions, the CV protocol, or n_splits — these
are what produced the reported diagnostic results.

Ported from scripts/stage1_noise_channel/7_stage_bottleneck_noise.py.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

logger = logging.getLogger(__name__)

SCALAR_FEATURES = ["mean_abs", "std", "energy", "max_abs", "kurtosis", "n_pixels"]
N_FOLDS = 5


# ── feature extraction ────────────────────────────────────────────────────────

def extract_scalar_features(noise_tensor: torch.Tensor) -> dict:
    """Extract 6 hand-crafted scalar features from a noise crop."""
    n = noise_tensor.numpy().astype(np.float64)
    abs_n = np.abs(n)
    flat = n.flatten()
    mean_val = flat.mean()
    std_val  = flat.std()
    return {
        "mean_abs": float(abs_n.mean()),
        "std":      float(std_val),
        "energy":   float((n ** 2).mean()),
        "max_abs":  float(abs_n.max()),
        "kurtosis": float(((flat - mean_val) ** 4).mean() /
                          (std_val ** 4 + 1e-12) - 3.0),
        "n_pixels": float(n.shape[-1] * n.shape[-2]),
    }


def load_noise(path: str) -> Optional[torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None


# ── TinyConvNet (levels 6 and 7) ─────────────────────────────────────────────

class TinyConvNet(nn.Module):
    """Minimal 2-layer ConvNet (~10 K params)."""

    def __init__(self, in_ch: int = 1, use_instance_norm: bool = False):
        super().__init__()
        self.use_in = use_instance_norm
        if use_instance_norm:
            self.inorm = nn.InstanceNorm2d(in_ch, affine=True)
        self.conv1 = nn.Conv2d(in_ch, 16, 5, stride=4, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_in:
            x = self.inorm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.fc(self.pool(x).flatten(1))


class _NoisePtDataset(Dataset):
    def __init__(self, paths: List[str], labels: torch.Tensor,
                 target_size: int = 128, in_ch: int = 1):
        self.paths = paths
        self.labels = labels
        self.target_size = target_size
        self.in_ch = in_ch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        noise = load_noise(self.paths[idx])
        if noise is None:
            noise = torch.zeros(self.in_ch, self.target_size, self.target_size)
        noise = F.interpolate(
            noise.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        return noise, self.labels[idx]


def _train_tiny_conv(
    train_paths: List[str], train_labels: torch.Tensor,
    val_paths:   List[str], val_labels:   torch.Tensor,
    in_ch: int = 1, use_instance_norm: bool = False,
    epochs: int = 10, lr: float = 1e-3,
) -> float:
    train_ds = _NoisePtDataset(train_paths, train_labels, in_ch=in_ch)
    val_ds   = _NoisePtDataset(val_paths,   val_labels,   in_ch=in_ch)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

    model = TinyConvNet(in_ch=in_ch, use_instance_norm=use_instance_norm)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch_noise, batch_labels in train_loader:
            optimizer.zero_grad()
            criterion(model(batch_noise), batch_labels).backward()
            optimizer.step()

    model.eval()
    all_scores: list = []
    all_labels: list = []
    with torch.no_grad():
        for batch_noise, batch_labels in val_loader:
            scores = torch.softmax(model(batch_noise), 1)[:, 1].numpy()
            all_scores.extend(scores)
            all_labels.extend(batch_labels.numpy())

    return float(roc_auc_score(all_labels, all_scores))


# ── main diagnostic function ──────────────────────────────────────────────────

def run_bottleneck_diagnostic(
    manifest_csv: str,
    output_dir: str,
    n_samples: Optional[int] = 3000,
    split: Optional[str] = "test",
    noise_col: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all 7 levels and return a tidy DataFrame of results.

    Columns: level, name, fold_aucs (list), mean_auc, std_auc.

    Parameters
    ----------
    manifest_csv : path to the manifest CSV (must contain noise_crop_path)
    output_dir   : where to write bottleneck_results.csv
    n_samples    : sub-sample limit (None = use all); default 3000 matches thesis
    split        : filter to this split column value; None = use all rows
    noise_col    : override column name for noise paths (auto-detected if None)
    seed         : random seed for sub-sampling and sklearn CV
    """
    import os
    import random

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(manifest_csv)
    if split and "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)

    if noise_col is None:
        for col in ("noise_crop_path", "noise_native_path"):
            if col in df.columns:
                noise_col = col
                break
    if noise_col is None:
        raise ValueError("No noise column found in manifest")

    valid = (df[noise_col].notna() & (df[noise_col] != "")
             & df[noise_col].apply(
                 lambda p: os.path.isfile(str(p)) if isinstance(p, str) and p else False))
    df = df[valid].reset_index(drop=True)

    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    logger.info(f"Bottleneck diagnostic: {len(df)} samples "
                f"({(df['label']==1).sum()} real, {(df['label']==0).sum()} fake)")

    # Load features
    features, noise_paths, labels_list = [], [], []
    for _, row in df.iterrows():
        path = str(row[noise_col])
        noise = load_noise(path)
        if noise is None:
            continue
        features.append(extract_scalar_features(noise))
        noise_paths.append(path)
        labels_list.append(int(row["label"]))

    X = np.array([[f[k] for k in SCALAR_FEATURES] for f in features])
    y = np.array(labels_list, dtype=np.int32)
    paths_arr = noise_paths

    sample_noise = load_noise(paths_arr[0])
    in_ch = sample_noise.shape[0] if sample_noise is not None else 1

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    results: list = []

    def _add(level: int, name: str, fold_aucs: list) -> None:
        mean_auc = float(np.mean(fold_aucs))
        std_auc  = float(np.std(fold_aucs))
        logger.info(f"  L{level} {name}: AUC={mean_auc:.4f} ± {std_auc:.4f}")
        results.append({
            "level": level, "name": name,
            "fold_aucs": fold_aucs,
            "mean_auc": mean_auc, "std_auc": std_auc,
        })

    # ── L1: logistic regression on mean_abs ──────────────────────────────
    aucs = []
    for tr_idx, vl_idx in skf.split(X, y):
        sc = StandardScaler().fit(X[tr_idx, 0:1])
        clf = LogisticRegression(max_iter=1000).fit(
            sc.transform(X[tr_idx, 0:1]), y[tr_idx])
        aucs.append(roc_auc_score(y[vl_idx],
            clf.predict_proba(sc.transform(X[vl_idx, 0:1]))[:, 1]))
    _add(1, "logreg_mean_abs", aucs)

    # ── L2: logistic regression on 6 scalar features ─────────────────────
    aucs = []
    for tr_idx, vl_idx in skf.split(X, y):
        sc = StandardScaler().fit(X[tr_idx])
        clf = LogisticRegression(max_iter=1000).fit(
            sc.transform(X[tr_idx]), y[tr_idx])
        aucs.append(roc_auc_score(y[vl_idx],
            clf.predict_proba(sc.transform(X[vl_idx]))[:, 1]))
    _add(2, "logreg_6feat", aucs)

    # ── L3: MLP (6→32→16→2) on 6 features ───────────────────────────────
    aucs = []
    for tr_idx, vl_idx in skf.split(X, y):
        sc = StandardScaler().fit(X[tr_idx])
        Xtr_t = torch.tensor(sc.transform(X[tr_idx]), dtype=torch.float32)
        ytr_t = torch.tensor(y[tr_idx], dtype=torch.long)
        Xvl_t = torch.tensor(sc.transform(X[vl_idx]), dtype=torch.float32)

        mlp = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 2),
        )
        opt  = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                            batch_size=64, shuffle=True)
        mlp.train()
        for _ in range(30):
            for xb, yb in loader:
                opt.zero_grad()
                crit(mlp(xb), yb).backward()
                opt.step()

        mlp.eval()
        with torch.no_grad():
            probs = torch.softmax(mlp(Xvl_t), 1)[:, 1].numpy()
        aucs.append(roc_auc_score(y[vl_idx], probs))
    _add(3, "mlp_6feat", aucs)

    # ── L4: logistic regression on InstanceNorm(noise) pooled 2×2 ────────
    inorm = nn.InstanceNorm2d(in_ch, affine=False)
    pool_feats = []
    for p in paths_arr:
        noise = load_noise(p)
        if noise is None:
            pool_feats.append(np.zeros(in_ch * 4))
            continue
        normed = inorm(noise.unsqueeze(0))
        pool_feats.append(F.adaptive_avg_pool2d(normed, 2).flatten().numpy())
    X_in = np.stack(pool_feats)

    aucs = []
    for tr_idx, vl_idx in skf.split(X_in, y):
        sc = StandardScaler().fit(X_in[tr_idx])
        clf = LogisticRegression(max_iter=1000).fit(
            sc.transform(X_in[tr_idx]), y[tr_idx])
        aucs.append(roc_auc_score(y[vl_idx],
            clf.predict_proba(sc.transform(X_in[vl_idx]))[:, 1]))
    _add(4, "logreg_inorm_pool", aucs)

    # ── L5: logistic regression on raw noise pooled 2×2 ──────────────────
    raw_feats = []
    for p in paths_arr:
        noise = load_noise(p)
        if noise is None:
            raw_feats.append(np.zeros(in_ch * 4))
            continue
        raw_feats.append(
            F.adaptive_avg_pool2d(noise.unsqueeze(0), 2).flatten().numpy())
    X_raw = np.stack(raw_feats)

    aucs = []
    for tr_idx, vl_idx in skf.split(X_raw, y):
        sc = StandardScaler().fit(X_raw[tr_idx])
        clf = LogisticRegression(max_iter=1000).fit(
            sc.transform(X_raw[tr_idx]), y[tr_idx])
        aucs.append(roc_auc_score(y[vl_idx],
            clf.predict_proba(sc.transform(X_raw[vl_idx]))[:, 1]))
    _add(5, "logreg_raw_pool", aucs)

    # ── L6: TinyConvNet on raw noise ──────────────────────────────────────
    aucs = []
    for tr_idx, vl_idx in skf.split(np.zeros(len(y)), y):
        tr_labels = torch.tensor(y[tr_idx], dtype=torch.long)
        vl_labels = torch.tensor(y[vl_idx], dtype=torch.long)
        aucs.append(_train_tiny_conv(
            [paths_arr[i] for i in tr_idx], tr_labels,
            [paths_arr[i] for i in vl_idx], vl_labels,
            in_ch=in_ch, use_instance_norm=False,
        ))
    _add(6, "tiny_convnet_raw", aucs)

    # ── L7: TinyConvNet on InstanceNorm(noise) ────────────────────────────
    aucs = []
    for tr_idx, vl_idx in skf.split(np.zeros(len(y)), y):
        tr_labels = torch.tensor(y[tr_idx], dtype=torch.long)
        vl_labels = torch.tensor(y[vl_idx], dtype=torch.long)
        aucs.append(_train_tiny_conv(
            [paths_arr[i] for i in tr_idx], tr_labels,
            [paths_arr[i] for i in vl_idx], vl_labels,
            in_ch=in_ch, use_instance_norm=True,
        ))
    _add(7, "tiny_convnet_inorm", aucs)

    result_df = pd.DataFrame([
        {"level": r["level"], "name": r["name"],
         "mean_auc": r["mean_auc"], "std_auc": r["std_auc"]}
        for r in results
    ])
    csv_path = f"{output_dir}/bottleneck_results.csv"
    result_df.to_csv(csv_path, index=False)
    logger.info(f"Wrote {csv_path}")
    return result_df
