"""
Config-driven MTCNN face-crop extraction for all four datasets.

Each dataset has its own extraction strategy (FF++ is video-level with actor
splits; CelebDF is video-level without splits; DFDC has a metadata.json for
splits; DFF is image-level).  A single config YAML selects the strategy.

Output for every dataset:
  <output_dir>/
    face_crops/
      train/   val/   test/   (or just all/ for datasets without splits)
    manifest.csv              (canonical column names — see manifest.py)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.data.manifest import MANIFEST_COLUMNS
from src.data.mtcnn_utils import (
    build_mtcnn,
    crop_with_padding,
    detect_faces_batch,
    find_first_face_frame,
    pick_best_face,
)
from src.data.ffpp_splits import (
    assign_split,
    create_splits,
    get_method_from_path,
    parse_filename as ffpp_parse_filename,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# DFF folder → (label, method)
# ──────────────────────────────────────────────────────────────────────────────

DFF_FOLDER_CONFIG: Dict[str, Tuple[int, str]] = {
    "inpainting": (1, "deepfakeface-inpainting"),
    "insight":    (1, "deepfakeface-insight"),
    "text2img":   (1, "deepfakeface-text2img"),
    "wiki":       (0, "deepfakeface-wiki"),
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ──────────────────────────────────────────────────────────────────────────────
# Shared video processing kernel
# ──────────────────────────────────────────────────────────────────────────────

def _process_video(
    video_path: str,
    split: str,
    label: int,
    method: str,
    video_id: str,
    output_dir: str,
    frames_per_video: int,
    padding_ratio: float,
    mtcnn,
    min_conf: float,
    batch_size: int = 8,
) -> List[dict]:
    """Extract face crops from one video and return manifest rows.

    Returns a list of dicts, one per saved crop.  Empty list if no face found.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    first_idx = find_first_face_frame(cap, mtcnn, max_check=60,
                                      batch_size=batch_size, min_conf=min_conf)
    if first_idx is None:
        cap.release()
        return []

    remaining  = total_frames - first_idx
    n_extract  = min(frames_per_video, remaining)
    if n_extract <= 0:
        cap.release()
        return []

    step = max(remaining // n_extract, 1)

    crop_dir = os.path.join(output_dir, "face_crops", split)
    os.makedirs(crop_dir, exist_ok=True)

    frames_bgr: list = []
    frames_rgb: list = []
    frame_idxs: list = []

    for i in range(n_extract):
        fidx = first_idx + i * step
        if fidx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        frames_bgr.append(frame)
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idxs.append(fidx)

    cap.release()

    if not frames_rgb:
        return []

    det_results = detect_faces_batch(mtcnn, frames_rgb, batch_size=batch_size)

    rows: List[dict] = []
    for frame_bgr, fidx, (boxes, probs) in zip(frames_bgr, frame_idxs, det_results):
        result = pick_best_face(boxes, probs, min_conf=min_conf)
        if result is None:
            continue

        box, conf = result
        crop, x1, y1, x2, y2 = crop_with_padding(frame_bgr, box, padding_ratio)
        if crop is None:
            continue

        img_name  = f"{video_id}_frame{fidx:04d}.jpg"
        crop_path = os.path.join(crop_dir, img_name)
        cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        rows.append({
            "filename":      img_name,
            "face_crop_path": crop_path,
            "label":          label,
            "split":          split,
            "method":         method,
            "video_id":       video_id,
            "frame_idx":      fidx,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence":     round(conf, 4),
            "dataset":        "",   # filled by caller
            "source":         "",   # filled by caller
        })

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# FF++
# ──────────────────────────────────────────────────────────────────────────────

def _collect_ffpp_videos(data_dir: str, splits: dict) -> Dict[str, list]:
    """Walk the FF++ directory tree and return {split: [(path, label, method)]}."""
    by_split: Dict[str, list] = {"train": [], "val": [], "test": []}

    search = [
        (os.path.join(data_dir, "original_sequences", "actors"),  0),
        (os.path.join(data_dir, "original_sequences", "youtube"), 0),
        (os.path.join(data_dir, "manipulated_sequences"),         1),
    ]

    for root_dir, label in search:
        if not os.path.isdir(root_dir):
            continue
        for root, _, files in os.walk(root_dir):
            for f in files:
                if not f.endswith(".mp4"):
                    continue
                vpath  = os.path.join(root, f)
                split  = assign_split(vpath, splits)
                method = get_method_from_path(vpath)
                if split:
                    by_split[split].append((vpath, label, method))

    return by_split


def extract_ffpp(cfg: dict, output_dir: str) -> str:
    """Extract FF++ face crops and write manifest.csv.  Returns manifest path."""
    data_dir         = cfg["data_dir"]
    frames_per_video = cfg.get("frames_per_video", 20)
    padding_ratio    = cfg.get("padding_ratio", 0.3)
    min_conf         = cfg.get("min_conf", 0.85)
    batch_size       = cfg.get("batch_size", 8)
    dataset_name     = cfg.get("dataset_name", "ff++")

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn  = build_mtcnn(device)
    splits = create_splits()

    logger.info("Collecting FF++ videos from %s", data_dir)
    by_split = _collect_ffpp_videos(data_dir, splits)
    for sp in ("train", "val", "test"):
        logger.info("  %s: %d videos", sp, len(by_split[sp]))

    write_header = not os.path.exists(manifest_path)
    stats = {"saved": 0, "no_face": 0}

    with open(manifest_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()

        for split in ("train", "val", "test"):
            for vpath, label, method in tqdm(by_split[split],
                                             desc=f"ff++ {split}"):
                video_id = os.path.splitext(os.path.basename(vpath))[0]
                rows = _process_video(
                    video_path=vpath, split=split, label=label,
                    method=method, video_id=video_id,
                    output_dir=output_dir,
                    frames_per_video=frames_per_video,
                    padding_ratio=padding_ratio, mtcnn=mtcnn,
                    min_conf=min_conf, batch_size=batch_size,
                )
                for row in rows:
                    row["dataset"] = dataset_name
                    row["source"]  = "original" if label == 0 else method
                    writer.writerow(row)
                if rows:
                    stats["saved"] += len(rows)
                else:
                    stats["no_face"] += 1

    logger.info("FF++ done: %d crops saved, %d videos with no face",
                stats["saved"], stats["no_face"])
    return manifest_path


# ──────────────────────────────────────────────────────────────────────────────
# CelebDF v2
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=10_000)
def _parse_celebdf_filename(filename: str) -> dict:
    basename = os.path.splitext(filename)[0]

    # YouTube-real: 5-digit numeric
    if "_" not in basename and basename.isdigit() and len(basename) == 5:
        return {"type": "real", "video_id": basename, "source": "youtube-real"}

    parts = basename.split("_")
    if len(parts) == 2:
        try:
            return {
                "type":     "real",
                "actor_id": int(parts[0][2:]),
                "scene_id": parts[1],
                "source":   "celeb-real",
            }
        except (ValueError, IndexError):
            pass
    elif len(parts) == 3:
        try:
            return {
                "type":            "fake",
                "source_actor_id": int(parts[0][2:]),
                "target_actor_id": int(parts[1][2:]),
                "scene_id":        parts[2],
                "source":          "celeb-synthesis",
            }
        except (ValueError, IndexError):
            pass

    raise ValueError(f"Cannot parse CelebDF filename: {filename!r}")


def _get_celebdf_method(video_path: str) -> str:
    parts = video_path.replace("\\", "/").split("/")
    for folder in ("Celeb-real", "YouTube-real", "Celeb-synthesis"):
        if folder in parts:
            return folder.lower()
    return "unknown"


def extract_celebdf(cfg: dict, output_dir: str) -> str:
    """Extract CelebDF v2 face crops and write manifest.csv."""
    data_dir         = cfg["data_dir"]
    frames_per_video = cfg.get("frames_per_video", 60)
    padding_ratio    = cfg.get("padding_ratio", 0.25)
    min_conf         = cfg.get("min_conf", 0.80)
    batch_size       = cfg.get("batch_size", 8)
    dataset_name     = cfg.get("dataset_name", "celebdf")
    # CelebDF has no actor-disjoint splits; all crops go to split="all"
    split            = cfg.get("split", "all")

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn  = build_mtcnn(device)

    video_paths: list = []
    for folder_name, label in [("Celeb-real", 0), ("YouTube-real", 0),
                                ("Celeb-synthesis", 1)]:
        folder = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder):
            logger.warning("CelebDF folder not found: %s", folder)
            continue
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".mp4"):
                    video_paths.append((os.path.join(root, f), label))

    logger.info("CelebDF: %d videos found", len(video_paths))

    write_header = not os.path.exists(manifest_path)
    stats = {"saved": 0, "no_face": 0}

    with open(manifest_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()

        for vpath, label in tqdm(video_paths, desc="celebdf"):
            video_id = os.path.splitext(os.path.basename(vpath))[0]
            method   = _get_celebdf_method(vpath)
            rows = _process_video(
                video_path=vpath, split=split, label=label,
                method=method, video_id=video_id,
                output_dir=output_dir,
                frames_per_video=frames_per_video,
                padding_ratio=padding_ratio, mtcnn=mtcnn,
                min_conf=min_conf, batch_size=batch_size,
            )
            for row in rows:
                row["dataset"] = dataset_name
                row["source"]  = method
                writer.writerow(row)
            if rows:
                stats["saved"] += len(rows)
            else:
                stats["no_face"] += 1

    logger.info("CelebDF done: %d crops saved, %d videos with no face",
                stats["saved"], stats["no_face"])
    return manifest_path


# ──────────────────────────────────────────────────────────────────────────────
# DFDC (test partition)
# ──────────────────────────────────────────────────────────────────────────────

def _load_dfdc_metadata(dfdc_dir: str) -> dict:
    meta_path = os.path.join(dfdc_dir, "metadata.json")
    with open(meta_path) as f:
        return json.load(f)


def _create_dfdc_splits(meta: dict, val_ratio: float = 0.15,
                        test_ratio: float = 0.15, seed: int = 42) -> dict:
    """Deterministic 70/15/15 split that keeps fake videos with their originals."""
    real_videos = [k for k, v in meta.items() if v["label"] == "REAL"]
    rng = random.Random(seed)
    rng.shuffle(real_videos)

    n      = len(real_videos)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    test_set  = set(real_videos[:n_test])
    val_set   = set(real_videos[n_test : n_test + n_val])

    split_map: dict = {}
    for name, info in meta.items():
        if info["label"] == "REAL":
            if name in test_set:
                split_map[name] = "test"
            elif name in val_set:
                split_map[name] = "val"
            else:
                split_map[name] = "train"
        else:
            original = info.get("original")
            if original and original in split_map:
                split_map[name] = split_map[original]
            else:
                r = rng.random()
                if r < test_ratio:
                    split_map[name] = "test"
                elif r < test_ratio + val_ratio:
                    split_map[name] = "val"
                else:
                    split_map[name] = "train"

    return split_map


def extract_dfdc(cfg: dict, output_dir: str) -> str:
    """Extract DFDC face crops and write manifest.csv."""
    data_dir         = cfg["data_dir"]
    frames_per_video = cfg.get("frames_per_video", 60)
    padding_ratio    = cfg.get("padding_ratio", 0.25)
    min_conf         = cfg.get("min_conf", 0.80)
    batch_size       = cfg.get("batch_size", 4)
    dataset_name     = cfg.get("dataset_name", "dfdc")

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn  = build_mtcnn(device)

    meta      = _load_dfdc_metadata(data_dir)
    split_map = _create_dfdc_splits(meta)

    logger.info("DFDC: %d videos in metadata.json", len(meta))

    write_header = not os.path.exists(manifest_path)
    stats = {"saved": 0, "no_face": 0}

    with open(manifest_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()

        for vid_file, info in tqdm(meta.items(), desc="dfdc"):
            video_path = os.path.join(data_dir, vid_file)
            if not os.path.exists(video_path):
                logger.warning("DFDC video not found: %s", video_path)
                continue

            label    = 1 if info["label"] == "FAKE" else 0
            split    = split_map.get(vid_file, "train")
            method   = "dfdc-fake" if label == 1 else "dfdc-real"
            video_id = os.path.splitext(vid_file)[0]

            rows = _process_video(
                video_path=video_path, split=split, label=label,
                method=method, video_id=video_id,
                output_dir=output_dir,
                frames_per_video=frames_per_video,
                padding_ratio=padding_ratio, mtcnn=mtcnn,
                min_conf=min_conf, batch_size=batch_size,
            )
            for row in rows:
                row["dataset"] = dataset_name
                row["source"]  = method
                writer.writerow(row)
            if rows:
                stats["saved"] += len(rows)
            else:
                stats["no_face"] += 1

    logger.info("DFDC done: %d crops saved, %d videos with no face",
                stats["saved"], stats["no_face"])
    return manifest_path


# ──────────────────────────────────────────────────────────────────────────────
# DeepFakeFace (image-based, no videos)
# ──────────────────────────────────────────────────────────────────────────────

def _dff_discover(data_dir: str) -> List[dict]:
    """Return all image records from the DFF directory tree."""
    records = []
    for folder_name, (label, method) in DFF_FOLDER_CONFIG.items():
        folder = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Expected DFF sub-folder: {folder}")
        for sub in sorted(os.listdir(folder)):
            sub_path = os.path.join(folder, sub)
            if not os.path.isdir(sub_path):
                continue
            for fname in sorted(os.listdir(sub_path)):
                if os.path.splitext(fname)[1].lower() not in _IMAGE_EXTS:
                    continue
                records.append({
                    "filepath": os.path.join(sub_path, fname),
                    "filename": fname,
                    "folder":   folder_name,
                    "label":    label,
                    "method":   method,
                })
    return records


def _dff_create_splits(records: list, val_ratio: float = 0.15,
                       test_ratio: float = 0.15, seed: int = 42) -> list:
    rng = random.Random(seed)
    by_folder: Dict[str, list] = {}
    for r in records:
        by_folder.setdefault(r["folder"], []).append(r)

    for items in by_folder.values():
        rng.shuffle(items)
        n      = len(items)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        for i, item in enumerate(items):
            if i < n_test:
                item["split"] = "test"
            elif i < n_test + n_val:
                item["split"] = "val"
            else:
                item["split"] = "train"

    return records


def extract_dff(cfg: dict, output_dir: str) -> str:
    """Extract DeepFakeFace face crops (from still images) and write manifest.csv."""
    data_dir      = cfg["data_dir"]
    padding_ratio = cfg.get("padding_ratio", 0.25)
    min_conf      = cfg.get("min_conf", 0.80)
    dataset_name  = cfg.get("dataset_name", "dff")

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn  = build_mtcnn(device)

    records = _dff_discover(data_dir)
    records = _dff_create_splits(records)
    logger.info("DFF: %d source images", len(records))

    write_header = not os.path.exists(manifest_path)
    stats = {"saved": 0, "no_face": 0}

    with open(manifest_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()

        for rec in tqdm(records, desc="dff"):
            img_bgr = cv2.imread(rec["filepath"])
            if img_bgr is None:
                stats["no_face"] += 1
                continue

            split  = rec["split"]
            label  = rec["label"]
            method = rec["method"]
            stem   = os.path.splitext(rec["filename"])[0]

            from PIL import Image as _PIL
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            from src.data.mtcnn_utils import detect_faces_batch as _dfb

            det = _dfb(mtcnn, [img_rgb], batch_size=1)
            boxes, probs = det[0]
            result = pick_best_face(boxes, probs, min_conf=min_conf)

            if result is None:
                stats["no_face"] += 1
                continue

            box, conf = result
            crop, x1, y1, x2, y2 = crop_with_padding(
                img_bgr, box, padding_ratio, target_size=299
            )
            if crop is None:
                stats["no_face"] += 1
                continue

            crop_dir  = os.path.join(output_dir, "face_crops", split)
            os.makedirs(crop_dir, exist_ok=True)
            img_name  = stem + ".jpg"
            crop_path = os.path.join(crop_dir, img_name)
            cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            writer.writerow({
                "filename":       img_name,
                "face_crop_path": crop_path,
                "label":          label,
                "split":          split,
                "method":         method,
                "video_id":       stem,
                "frame_idx":      0,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence":     round(conf, 4),
                "dataset":        dataset_name,
                "source":         rec["folder"],
            })
            stats["saved"] += 1

    logger.info("DFF done: %d crops saved, %d images with no face",
                stats["saved"], stats["no_face"])
    return manifest_path


# ──────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ──────────────────────────────────────────────────────────────────────────────

_EXTRACTORS = {
    "ff++":    extract_ffpp,
    "celebdf": extract_celebdf,
    "dfdc":    extract_dfdc,
    "dff":     extract_dff,
}


def extract(cfg: dict) -> str:
    """Run extraction for the dataset named in cfg['dataset'] and return the manifest path."""
    dataset    = cfg["dataset"]
    output_dir = cfg["output_dir"]

    if dataset not in _EXTRACTORS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Choices: {list(_EXTRACTORS)}"
        )

    fn = _EXTRACTORS[dataset]
    logger.info("Starting extraction: dataset=%s  output_dir=%s", dataset, output_dir)
    t0 = time.time()
    manifest = fn(cfg, output_dir)
    elapsed  = time.time() - t0
    logger.info("Extraction complete in %.1fs → %s", elapsed, manifest)
    return manifest
