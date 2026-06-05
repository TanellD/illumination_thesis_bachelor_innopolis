"""
FF++ filename parsing and actor-disjoint train/val/test split assignment.

The split logic is load-bearing: splits were created once with seed=42 and
all reported numbers depend on the same assignment. Do not change the seed or
the pool sizes.

FF++ has two naming conventions depending on the video source:
  - "actors" source: filenames like "01__02__scene1__uid" or "01__uid" (real)
  - "youtube" source: 3-digit numeric real ("001"), 3+3 digit fake ("001_002")

Split sizes:
  Actors  : 20 train / 4 val / 4 test  (28 total actors in the dataset)
  YouTube : 700 train / 150 val / 150 test  (1000 total IDs 000-999)
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from typing import Dict, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Filename parsing
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=10_000)
def parse_filename(filename: str) -> dict:
    """Parse an FF++ video filename and return a dict of metadata.

    Raises ValueError if the filename does not match any known pattern.
    """
    basename = filename.split(".")[0]

    if "__" not in basename:
        if "_" in basename:
            parts = basename.split("_")
            if len(parts) == 2 and len(parts[0]) == 3 and len(parts[1]) == 3:
                return {
                    "type":    "deepfake",
                    "actor_1": parts[0],
                    "actor_2": parts[1],
                    "source":  "youtube",
                }
        elif basename.isdigit():
            if len(basename) == 3:
                return {"type": "real", "actor_1": basename, "source": "youtube"}
            elif len(basename) == 2:
                return {"type": "real", "actor_1": basename, "source": "actors"}
        raise ValueError(f"Cannot parse FF++ filename: {filename!r}")

    parts  = basename.split("__")
    actors = parts[0].split("_")
    result: dict = {"source": "actors"}

    if len(actors) == 2:
        result.update(type="deepfake", actor_1=actors[0], actor_2=actors[1])
    elif len(actors) == 1:
        result.update(type="real", actor_1=actors[0])
    else:
        raise ValueError(f"Cannot parse FF++ filename: {filename!r}")

    if len(parts) >= 2:
        result["scene"] = parts[1]
    if len(parts) >= 3:
        result["unique_id"] = parts[2]

    return result


@lru_cache(maxsize=1_000)
def get_method_from_path(video_path: str) -> str:
    """Extract the manipulation method name from the video path.

    Looks for 'manipulated_sequences/<method>/' in the path.
    Returns 'original' for real videos.
    """
    parts = video_path.replace("\\", "/").split("/")
    if "manipulated_sequences" in parts:
        idx = parts.index("manipulated_sequences")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "original"


# ──────────────────────────────────────────────────────────────────────────────
# Split creation  (seed=42, deterministic)
# ──────────────────────────────────────────────────────────────────────────────

def create_splits() -> Dict[str, set]:
    """Return the canonical actor-disjoint split pools (seed=42).

    This function reproduces exactly what the original ffpp_cropping_splitting.py
    did. Calling it twice with the same Python version gives the same result.
    """
    actor_ids   = [f"{i:02d}" for i in range(1, 29)]
    youtube_ids = [f"{i:03d}" for i in range(1000)]

    rng = random.Random(42)
    rng.shuffle(actor_ids)
    rng.shuffle(youtube_ids)

    return {
        "train_actors":   set(actor_ids[:20]),
        "val_actors":     set(actor_ids[20:24]),
        "test_actors":    set(actor_ids[24:28]),
        "train_youtube":  set(youtube_ids[:700]),
        "val_youtube":    set(youtube_ids[700:850]),
        "test_youtube":   set(youtube_ids[850:]),
    }


def assign_split(video_path: str, splits: Dict[str, set]) -> Optional[str]:
    """Return 'train', 'val', or 'test' for a video path, or None if unknown.

    For deepfakes, the split is determined by the *target* actor (actor_2),
    matching the original script behaviour.
    """
    try:
        info = parse_filename(os.path.basename(video_path))
    except ValueError:
        return None

    if info["type"] == "deepfake":
        target = info.get("actor_2", info["actor_1"])
    else:
        target = info["actor_1"]

    src = info["source"]

    if src == "actors":
        pools = [("train", "train_actors"), ("val", "val_actors"), ("test", "test_actors")]
    else:
        pools = [("train", "train_youtube"), ("val", "val_youtube"), ("test", "test_youtube")]

    for name, key in pools:
        if target in splits[key]:
            return name

    return None
