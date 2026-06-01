"""
src/robustness/grid.py
======================
Robustness perturbation grid runner (experiment E.*).

For each (model, dataset, perturbation_family, perturbation_value), runs
inference on the perturbed test split and records the five metric tuple.

Output: one long-form CSV per call with columns:
    model, dataset, perturbation_family, perturbation_value,
    auc, eer, fpr_at_tpr95, ece, brier, n_samples

The analysis layer (src/report/) reads this CSV; it never needs GPU access.

Usage:
    from src.robustness.grid import run_robustness_grid
    run_robustness_grid(model, loader_factory, output_csv, device)
"""
from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.eval.metrics import compute_metrics
from src.robustness.perturbations import (
    PERTURBATION_GRID,
    apply_perturbation,
)

logger = logging.getLogger(__name__)


def run_robustness_grid(
    model: nn.Module,
    loader_factory: Callable[[Optional[str]], object],
    model_name: str,
    dataset_name: str,
    output_csv: str,
    device: torch.device,
    perturbation_grid: Optional[List[Tuple]] = None,
    use_video_ids: bool = False,
) -> pd.DataFrame:
    """Evaluate a model over the full perturbation grid.

    Parameters
    ----------
    model          : trained nn.Module in eval mode
    loader_factory : callable() → DataLoader returning (img, label, *extra)
                     where img is a normalised [-1,1] tensor batch
    model_name     : used in the output CSV's 'model' column
    dataset_name   : used in the output CSV's 'dataset' column
    output_csv     : path to write the long-form results CSV
    device         : torch device for inference
    perturbation_grid : list of (family, param) tuples;
                        defaults to the canonical thesis grid
    use_video_ids  : if True, loader yields (img, label, video_id, source);
                     video_id/source are ignored (frame-level metrics only)

    Returns
    -------
    pd.DataFrame with one row per (family, param)
    """
    if perturbation_grid is None:
        perturbation_grid = PERTURBATION_GRID

    rows = []
    model.eval()
    model.to(device)

    for family, param in perturbation_grid:
        param_str = str(param)
        t0 = time.time()

        loader = loader_factory()
        all_scores: list = []
        all_labels: list = []

        with torch.no_grad():
            for batch in loader:
                imgs   = batch[0].to(device)
                labels = batch[1]
                imgs = apply_perturbation(imgs, family, param)
                out = model(imgs)
                # Handle both single-logit (BCEWithLogitsLoss) and two-class softmax
                if out.ndim == 2 and out.shape[1] == 2:
                    probs = torch.softmax(out, 1)[:, 1]
                elif out.ndim == 2 and out.shape[1] == 1:
                    probs = torch.sigmoid(out.squeeze(1))
                else:
                    probs = torch.sigmoid(out)
                all_scores.extend(probs.float().cpu().tolist())
                all_labels.extend(labels.tolist()
                                  if hasattr(labels, "tolist") else list(labels))

        s = np.array(all_scores,  dtype=np.float64)
        l = np.array(all_labels,  dtype=np.int32)
        m = compute_metrics(l, s)

        row = {
            "model":               model_name,
            "dataset":             dataset_name,
            "perturbation_family": family,
            "perturbation_value":  param_str,
            "auc":                 m["auc"],
            "eer":                 m["eer"],
            "fpr_at_tpr95":        m["fpr_at_tpr95"],
            "ece":                 m["ece"],
            "brier":               m["brier"],
            "n_samples":           m["n"],
            "elapsed_s":           round(time.time() - t0, 2),
        }
        rows.append(row)
        logger.info(
            f"  [{model_name}|{dataset_name}] {family}={param_str:6s}  "
            f"AUC={m['auc']:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote {output_csv} ({len(df)} rows)")
    return df
