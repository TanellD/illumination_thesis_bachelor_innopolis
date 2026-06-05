"""
src/train/loop.py
=================
Training loop, optimizer factory, and backbone-unfreezing callbacks for
Stage 1 models (RGB-Only, Residual-Only, Late-Fusion, StatNoise-Fusion,
ResAware-Fusion).

All hyperparameters match the thesis (§3.6):
  - AdamW with per-group LR multipliers (see build_optimizer)
  - 3-epoch linear warmup from 0.1 → 1.0
  - ReduceLROnPlateau (mode='max', patience=2, factor=0.5) tracking val F1-macro
  - Gradient clip at 1.0
  - Early stopping with patience=7
  - Progressive backbone unfreezing at epochs 3 and 6 (Late-Fusion, ResAware only)
  - Max 30 epochs

Do NOT change these values — they are what produced the reported results.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Seeds used in the five-seed ablation (B.1) and two fixed-fusion variants (B.4)
STAGE1_SEEDS = (42, 123, 456, 789, 1337)


# ── seed helper ───────────────────────────────────────────────────────────────

def set_all_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds deterministically."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── optimizer factory ─────────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    model_name: str,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Per-component AdamW with model-specific LR multipliers.

    LR multipliers (from training_models.py):
      RGB-Only:       backbone×0.1,  classifier×1.0
      Residual-Only:  all params   ×1.0  (lr=1e-4 hard-coded)
      Late-Fusion:    rgb×0.05, noise×1.0, classifier×0.5
      StatNoise-Fusion: rgb×0.005, noise_mlp×2.0, classifier×0.05
      ResAware-Fusion:  rgb×0.05,  noise×1.0,  classifier×0.5
    """
    name = model_name.lower().replace("-", "_").replace(" ", "_")

    if name == "rgb_only":
        backbone_p   = [p for n, p in model.named_parameters() if "classifier" not in n]
        classifier_p = [p for n, p in model.named_parameters() if "classifier"     in n]
        return torch.optim.AdamW([
            {"params": backbone_p,   "lr": lr * 0.1},
            {"params": classifier_p, "lr": lr},
        ], weight_decay=weight_decay)

    if name == "residual_only":
        return torch.optim.AdamW(model.parameters(), lr=1e-4,
                                  weight_decay=weight_decay)

    if name in ("late_fusion", "resaware_fusion"):
        rgb_p = [p for n, p in model.named_parameters() if "rgb_backbone"  in n]
        noi_p = [p for n, p in model.named_parameters()
                 if "noise_backbone" in n or "noise_norm" in n or "noise_fc" in n]
        cls_p = [p for n, p in model.named_parameters() if "classifier"    in n]
        return torch.optim.AdamW([
            {"params": rgb_p, "lr": lr * 0.05},
            {"params": noi_p, "lr": lr},
            {"params": cls_p, "lr": lr * 0.5},
        ], weight_decay=weight_decay)

    if name == "statnoise_fusion":
        rgb_p = [p for n, p in model.named_parameters() if "rgb_backbone" in n]
        noi_p = [p for n, p in model.named_parameters() if "noise_mlp"    in n]
        cls_p = [p for n, p in model.named_parameters() if "classifier"   in n]
        return torch.optim.AdamW([
            {"params": rgb_p, "lr": lr * 0.005},
            {"params": noi_p, "lr": lr * 2.0},
            {"params": cls_p, "lr": lr * 0.05},
        ], weight_decay=weight_decay)

    # Fallback: all params at base lr
    return torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=weight_decay)


# ── progressive backbone unfreezing ──────────────────────────────────────────

def maybe_unfreeze(model: nn.Module, model_name: str, epoch: int) -> None:
    """Unfreeze Xception blocks progressively for fusion models.

    epoch=3: unfreeze block10–block12
    epoch=6: unfreeze block4–block9

    Only applies to Late-Fusion and ResAware-Fusion (same as original script).
    """
    name = model_name.lower().replace("-", "_")
    if name not in ("late_fusion", "resaware_fusion"):
        return

    if epoch == 3:
        _unfreeze_blocks(model, ["block10", "block11", "block12"])
        logger.info(f"[{model_name}] unfroze Xception block10-12")
    elif epoch == 6:
        _unfreeze_blocks(model, [f"block{i}" for i in range(4, 10)])
        logger.info(f"[{model_name}] unfroze Xception block4-9")


def _unfreeze_blocks(model: nn.Module, block_names: list[str]) -> None:
    for name, param in model.named_parameters():
        if any(b in name for b in block_names):
            param.requires_grad = True


# ── training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_name: str = "",
    epoch: int = 0,
    batch_callback: Optional[Callable] = None,
) -> dict:
    """Single training epoch.  Returns dict: loss, f1_macro, accuracy."""
    model.train()
    total_loss = 0.0
    all_labels: list = []
    all_preds:  list = []

    # Models that accept a second positional tensor (precomputed noise crop).
    # ResidualOnly extracts noise internally — do NOT pass it as a second arg.
    _model_takes_noise_arg = model_name.lower().replace("-", "_").replace(" ", "_") in (
        "late_fusion", "statnoise_fusion", "resaware_fusion"
    )

    for batch in loader:
        # Unpack batch regardless of source dataset type.
        # NoiseCropDataset → (rgb, noise, label, vid)  [4 elements]
        # FaceCropDataset  → (rgb, label, vid)          [3 elements]
        if len(batch) == 4:
            rgb, noise, labels, _ = batch
            if _model_takes_noise_arg:
                inputs = (rgb.to(device), noise.to(device))
            else:
                inputs = (rgb.to(device),)
        elif len(batch) == 3 and isinstance(batch[0], torch.Tensor):
            rgb, labels, _ = batch
            inputs = (rgb.to(device),)
        else:
            inputs, labels = batch[:-1], batch[-1]

        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(*inputs) if len(inputs) > 1 else model(inputs[0])
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        if batch_callback is not None:
            batch_callback()

    n = len(loader)
    return {
        "loss":     total_loss / max(n, 1),
        "f1_macro": f1_score(all_labels, all_preds, average="macro",
                              zero_division=0),
        "accuracy": float(np.mean(np.array(all_preds) == np.array(all_labels))),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str = "",
) -> dict:
    """Validation pass.  Returns dict: loss, f1_macro, auc."""
    model.eval()
    total_loss = 0.0
    all_labels: list = []
    all_preds:  list = []
    all_probs:  list = []

    _model_takes_noise_arg = model_name.lower().replace("-", "_").replace(" ", "_") in (
        "late_fusion", "statnoise_fusion", "resaware_fusion"
    )

    for batch in loader:
        if len(batch) == 4:
            rgb, noise, labels, _ = batch
            if _model_takes_noise_arg:
                inputs = (rgb.to(device), noise.to(device))
            else:
                inputs = (rgb.to(device),)
        elif len(batch) == 3 and isinstance(batch[0], torch.Tensor):
            rgb, labels, _ = batch
            inputs = (rgb.to(device),)
        else:
            inputs, labels = batch[:-1], batch[-1]

        labels = labels.to(device)
        out = model(*inputs) if len(inputs) > 1 else model(inputs[0])
        total_loss += criterion(out, labels).item()
        probs = torch.softmax(out, 1)[:, 1].cpu().tolist()
        all_probs.extend(probs)
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    n = len(loader)
    try:
        auc = float(roc_auc_score(all_labels, all_probs)) \
              if len(set(all_labels)) > 1 else 0.5
    except Exception:
        auc = 0.5

    return {
        "loss":     total_loss / max(n, 1),
        "f1_macro": f1_score(all_labels, all_preds, average="macro",
                              zero_division=0),
        "auc":      auc,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    model_name: str,
    seed: int,
    epochs: int = 30,
    lr: float = 1e-3,
    warmup_epochs: int = 3,
    early_stop_patience: int = 7,
) -> tuple[nn.Module, list, list]:
    """Full training run for one model / one seed.

    Returns (trained_model, train_history, val_history).
    History items are dicts with keys: epoch, loss, f1_macro, [auc].
    """
    optimizer = build_optimizer(model, model_name, lr)
    criterion = nn.CrossEntropyLoss()

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    best_f1 = 0.0
    best_state: Optional[dict] = None
    patience_ctr = 0
    train_history: list = []
    val_history:   list = []

    for epoch in range(epochs):
        maybe_unfreeze(model, model_name, epoch)

        t = train_one_epoch(model, train_loader, optimizer, criterion, device,
                             model_name=model_name, epoch=epoch)
        v = evaluate(model, val_loader, criterion, device, model_name=model_name)

        if epoch < warmup_epochs:
            warmup_sched.step()
        else:
            plateau_sched.step(v["f1_macro"])

        train_history.append({"epoch": epoch + 1, **t,
                               "lr": optimizer.param_groups[0]["lr"]})
        val_history.append({"epoch": epoch + 1, **v})

        logger.info(
            f"[{model_name}|seed={seed}] ep {epoch+1:02d}  "
            f"train_loss={t['loss']:.4f}  val_f1={v['f1_macro']:.4f}  "
            f"val_auc={v['auc']:.4f}"
        )

        if v["f1_macro"] > best_f1:
            best_f1 = v["f1_macro"]
            best_state = {k: v2.clone() for k, v2 in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                logger.info(f"[{model_name}|seed={seed}] early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_history, val_history
