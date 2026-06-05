"""
experiments/c1_generate_tsbi.py
================================
Entry point for experiment C.1 — T-SBI generation.

Reads real FF++ videos, runs MTCNN face detection, selects illumination-delta
pairs, applies one of the five transfer modes, and writes JPEG crops + CSV.

CLI:
    thesis-c1-generate-tsbi --config configs/stage2/tsbi.yaml [--force]

Required config keys:
    data_dir:              path to FF++ video root
    output_dir:            where to write crops and tsbi_labels.csv
    per_video:             target T-SBI pairs per real video (default 10)
    splits:                list of splits to process (default [train])
    min_gap_sec:           minimum temporal gap in seconds (default 0.8)
    max_gap_sec:           maximum temporal gap in seconds (default 5.0)
    min_illum_delta_mean:  dL_mean threshold (default 6.0)
    min_illum_delta_std:   dL_std threshold (default 3.0)
    illum_relax:           allow best-effort fallback (default true)
    padding_ratio:         face crop padding (default 0.3)
    jpeg_quality_range:    [min, max] (default [75, 98])
    modes:                 list of transfer modes (default all five)
    seed:                  random seed (default 42)
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

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T-SBI generation")
    p.add_argument("--config", required=True)
    p.add_argument("--out-csv", default=None,
                   help="Also copy the output CSV to this flat path "
                        "(e.g. outputs/tsbi_labels.csv) for downstream scripts.")
    p.add_argument("--out-dir", default=None,
                   help="Override crop output directory.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t0 = time.time()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolution order: env var > config key > paths.yaml
    if "THESIS_FFPP_ROOT" in os.environ:
        data_dir = os.environ["THESIS_FFPP_ROOT"]
    elif "data_dir" in cfg:
        data_dir = cfg["data_dir"]
    else:
        try:
            from src.utils.paths import load_paths
            data_dir = str(load_paths().data.ff_plus_plus)
        except Exception:
            raise KeyError(
                "FF++ data_dir not found. Set THESIS_FFPP_ROOT env var, "
                "add 'data_dir:' to configs/stage2/tsbi.yaml, "
                "or set data.ff_plus_plus in paths.yaml"
            )
    output_dir = os.environ.get("THESIS_OUTPUT_ROOT",
                                cfg.get("output_dir", "outputs"))
    cfg_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()[:12]
    # Crop images go into content-hashed subdir; CSV also copied to flat path
    out_root = Path(args.out_dir) if args.out_dir else \
               Path(output_dir) / "c1_tsbi" / cfg_hash
    out_root.mkdir(parents=True, exist_ok=True)

    # Canonical CSV inside the hashed dir
    out_csv = str(out_root / "tsbi_labels.csv")
    # Flat copy path for downstream scripts (e.g. make manifests)
    flat_csv = args.out_csv or str(Path(output_dir) / "tsbi_labels.csv")

    if Path(out_csv).exists() and not args.force:
        logger.info(f"Output already exists at {out_csv}. Use --force to rerun.")
        sys.exit(0)

    from facenet_pytorch import MTCNN
    from src.tsbi.generator import (
        MODES, TSBI_CSV_COLS,
        select_pair, face_box_luma_stats,
        detect_landmarks_68, hull_mask_from_landmarks,
        ellipse_mask_on_box, align_from_box,
        crop_with_padding, boxes_compatible, box_xyxy_to_xywh,
        FACE_REGIONS, _HAS_MP, _FaceMeshSingleton,
    )
    from src.data.ffpp_splits import create_splits, assign_split

    seed       = cfg.get("seed", 42)
    per_video  = cfg.get("per_video", 10)
    splits     = cfg.get("splits", ["train"])
    min_gap    = cfg.get("min_gap_sec", 0.8)
    max_gap    = cfg.get("max_gap_sec", 5.0)
    min_dLm    = cfg.get("min_illum_delta_mean", 6.0)
    min_dLs    = cfg.get("min_illum_delta_std", 3.0)
    relax      = cfg.get("illum_relax", True)
    pad        = cfg.get("padding_ratio", 0.3)
    qmin, qmax = cfg.get("jpeg_quality_range", [75, 98])
    modes      = cfg.get("modes", list(MODES.keys()))
    no_lm      = cfg.get("no_landmarks", False)

    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, thresholds=[0.6, 0.7, 0.7],
                  post_process=True, device=device,
                  select_largest=False, min_face_size=20)

    ff_splits = create_splits()
    # Collect real video paths
    vid_paths = []
    for subdir in ("original_sequences/actors", "original_sequences/youtube"):
        root = os.path.join(data_dir, subdir)
        if not os.path.exists(root):
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                if not f.endswith(".mp4"):
                    continue
                vp = os.path.join(dirpath, f)
                sp = assign_split(vp, ff_splits)
                if sp in splits:
                    vid_paths.append((vp, sp))
    logger.info(f"Real videos to process: {len(vid_paths)}")

    master_rng = random.Random(seed)

    def scan_usable(cap, total, fps_):
        stride = max(1, int(fps_ // 3))
        good = []
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
                    if pp is None or pp < 0.85:
                        continue
                    x1, y1, x2, y2 = bb
                    a = (x2 - x1) * (y2 - y1)
                    if a > ba:
                        ba = a; best = bb
                if best is None:
                    continue
                stats = face_box_luma_stats(frame, best)
                if stats is None:
                    continue
                good.append((idx, best, stats))
            except Exception:
                continue
        return good

    total_written = 0
    total_relaxed = 0

    with open(out_csv, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(TSBI_CSV_COLS)

        for vp, sp in tqdm(vid_paths, desc="videos"):
            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                continue
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if total <= 0:
                cap.release(); continue

            usable = scan_usable(cap, total, fps)
            if len(usable) < 2:
                cap.release(); continue

            rng = random.Random(master_rng.random())
            wrote = 0; attempts = 0
            vstem = Path(vp).stem
            out_sub = out_root / sp
            out_sub.mkdir(parents=True, exist_ok=True)

            while wrote < per_video and attempts < per_video * 8:
                attempts += 1
                result = select_pair(usable, rng, fps,
                                     min_gap, max_gap, min_dLm, min_dLs,
                                     relax=relax)
                if result is None:
                    break
                src_f, tgt_f, dLm, dLs = result
                is_relaxed = not (dLm >= min_dLm or dLs >= min_dLs)

                s_idx, s_box_xyxy, _ = src_f
                t_idx, t_box_xyxy, _ = tgt_f

                cap.set(cv2.CAP_PROP_POS_FRAMES, s_idx)
                ok, s_frame = cap.read()
                if not ok:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, t_idx)
                ok, t_frame = cap.read()
                if not ok:
                    continue

                s_box = box_xyxy_to_xywh(s_box_xyxy)
                t_box = box_xyxy_to_xywh(t_box_xyxy)

                # Build mask
                mask = None
                mask_kind = "ellipse"
                if _HAS_MP and not no_lm:
                    lms = detect_landmarks_68(t_frame)
                    if lms is not None:
                        region = rng.choice(list(FACE_REGIONS.keys()))
                        mask = hull_mask_from_landmarks(
                            lms, t_frame.shape, region=region,
                            feather_ratio=rng.uniform(0.05, 0.10),
                        )
                        mask_kind = f"hull:{region}"
                if mask is None:
                    mask = ellipse_mask_on_box(
                        t_box, t_frame.shape,
                        feather_ratio=rng.uniform(0.18, 0.24),
                    )

                src_ref = align_from_box(s_frame, s_box, t_box, t_frame.shape)
                mode = rng.choice(modes)
                try:
                    fake_full = MODES[mode](t_frame, src_ref, mask)
                except Exception:
                    continue

                crop, bbox = crop_with_padding(fake_full, t_box_xyxy, pad=pad)
                if crop is None:
                    continue

                q = random.randint(qmin, qmax)
                fname = f"{vstem}_tsbi_{t_idx:06d}_{wrote:03d}.jpg"
                fpath = str(out_sub / fname)
                cv2.imwrite(fpath, crop, [cv2.IMWRITE_JPEG_QUALITY, q])

                x1, y1, x2, y2 = bbox
                w.writerow([
                    fname, fpath, 0, sp, "T-SBI",
                    x1, y1, x2, y2, 1.0,
                    s_idx, t_idx, mode, vp,
                    q, mask_kind,
                    round(dLm, 4), round(dLs, 4), int(is_relaxed),
                ])
                wrote += 1
                total_written += 1
                if is_relaxed:
                    total_relaxed += 1

            cap.release()

    logger.info(f"Written: {total_written} crops  "
                f"({total_relaxed} with relaxed illumination gate)")

    sidecar = {
        "config":    str(args.config),
        "cfg_hash":  cfg_hash,
        "n_written": total_written,
        "n_relaxed": total_relaxed,
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

    with open(out_root / "sidecar.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    try:
        _FaceMeshSingleton.cleanup()
    except Exception:
        pass

    # Copy to flat path so downstream scripts (make manifests) can find it
    if flat_csv != out_csv:
        import shutil
        Path(flat_csv).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_csv, flat_csv)
        logger.info(f"Copied CSV to {flat_csv}")

    logger.info(f"Done. Outputs in {out_root}")


if __name__ == "__main__":
    main()
