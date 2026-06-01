"""
src/eval/cache.py
=================
NPZ-backed per-(model, dataset) inference cache.

The cache stores per-frame predictions so the inference pass (which needs
a GPU) can be run once and the analysis pass (CPU-only) can read from it
without re-running the model.

Key convention:
    <cache_root>/inference/<safe_model>_<safe_dataset>.npz
where safe_key() replaces "/" → "-", " " → "_", "+" → "p".

Each NPZ file stores arrays:
    scores   : (N,) float32 — p(real) per frame
    labels   : (N,) int32   — 0 fake / 1 real
    video_ids: (N,) object  — video identifier strings
    sources  : (N,) object  — source-tag strings
    methods  : (N,) object  — manipulation method strings
    paths    : (N,) object  — face-crop file paths

Usage
-----
>>> cache = DiskCache("/tmp/cache")
>>> cache.save_inference("RGB-Only", "ff++", scores, labels, ...)
>>> result = cache.load_inference("RGB-Only", "ff++")  # None if missing
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


def safe_key(s: str) -> str:
    return s.replace("/", "-").replace(" ", "_").replace("+", "p")


class DiskCache:
    """NPZ-backed cache rooted at *root_dir*."""

    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir)
        self._inference_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _inference_dir(self) -> Path:
        return self.root / "inference"

    def _inference_path(self, model: str, dataset: str) -> Path:
        return self._inference_dir / f"{safe_key(model)}_{safe_key(dataset)}.npz"

    # ── inference ──────────────────────────────────────────────────────────────

    def save_inference(
        self,
        model: str,
        dataset: str,
        scores: np.ndarray,
        labels: np.ndarray,
        video_ids: Optional[np.ndarray] = None,
        sources: Optional[np.ndarray] = None,
        methods: Optional[np.ndarray] = None,
        paths: Optional[np.ndarray] = None,
    ) -> Path:
        """Write inference arrays to the cache.  Returns the file path."""
        p = self._inference_path(model, dataset)
        arrays: dict = {
            "scores": np.asarray(scores, dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.int32),
        }
        if video_ids is not None:
            arrays["video_ids"] = np.asarray(video_ids, dtype=object)
        if sources is not None:
            arrays["sources"] = np.asarray(sources, dtype=object)
        if methods is not None:
            arrays["methods"] = np.asarray(methods, dtype=object)
        if paths is not None:
            arrays["paths"] = np.asarray(paths, dtype=object)
        np.savez(str(p), **arrays)
        return p

    def load_inference(self, model: str, dataset: str) -> Optional[dict]:
        """Return a dict of arrays, or None if not cached."""
        p = self._inference_path(model, dataset)
        if not p.exists():
            return None
        with np.load(str(p), allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    def has_inference(self, model: str, dataset: str) -> bool:
        return self._inference_path(model, dataset).exists()

    def delete_inference(self, model: str, dataset: str) -> bool:
        """Delete a cached NPZ.  Returns True if the file existed."""
        p = self._inference_path(model, dataset)
        if p.exists():
            p.unlink()
            return True
        return False

    def list_inference(self) -> list[tuple[str, str]]:
        """List all (model, dataset) pairs present in the cache.

        Only pairs whose filenames can be split at the first underscore are
        returned; malformed filenames are silently skipped.
        """
        pairs = []
        for f in sorted(self._inference_dir.glob("*.npz")):
            stem = f.stem
            # We stored as safe_key(model)_safe_key(dataset). Reverse lookup
            # is ambiguous (model names can contain underscores) so we just
            # return the raw stem split on first underscore.
            pairs.append(tuple(stem.split("_", 1)))
        return pairs
