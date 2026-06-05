"""
PyTorch Dataset classes for the thesis pipeline.

FaceCropDataset   — loads RGB face crops; used by Stage 2 training.
NoiseCropDataset  — loads both RGB crops and precomputed noise .pt files;
                    used by Stage 1 training (ResidualOnly, LateFusion).

Both classes accept a manifest DataFrame (produced by load_manifest) and
a torchvision transform.  They are deterministic for a given manifest.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.manifest import load_manifest


class FaceCropDataset(Dataset):
    """RGB face crops for Stage 2 / evaluation.

    Returns:
        (image_tensor, label, video_id)
        where image_tensor is a float32 CHW tensor after transform.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        transform: Optional[Callable] = None,
        split: Optional[str] = None,
    ) -> None:
        if split is not None and not split in ['all', 'None']:
            manifest = manifest[manifest["split"] == split].reset_index(drop=True)
        self.df        = manifest
        self.transform = transform

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        transform: Optional[Callable] = None,
        split: Optional[str] = None,
        check_files_exist: bool = False,
    ) -> "FaceCropDataset":
        df = load_manifest(csv_path, check_files_exist=check_files_exist)
        return cls(df, transform=transform, split=split)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row   = self.df.iloc[idx]
        img   = Image.open(str(row["face_crop_path"])).convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            img = self.transform(img)

        video_id = str(row["video_id"])
        return img, label, video_id


class NoiseCropDataset(Dataset):
    """RGB face crops + precomputed noise maps for Stage 1.

    Returns:
        (rgb_tensor, noise_tensor, label, video_id)
        where rgb_tensor is float32 CHW and noise_tensor is [1, 299, 299] float32.

    Requires noise_crop_path column in the manifest (written by noise precompute).
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        rgb_transform: Optional[Callable] = None,
        split: Optional[str] = None,
    ) -> None:
        if split is not None:
            manifest = manifest[manifest["split"] == split].reset_index(drop=True)

        if "noise_crop_path" not in manifest.columns:
            raise ValueError(
                "Manifest has no noise_crop_path column. "
                "Run noise precompute (a5_noise_precompute) first."
            )

        self.df            = manifest
        self.rgb_transform = rgb_transform

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        rgb_transform: Optional[Callable] = None,
        split: Optional[str] = None,
        check_files_exist: bool = False,
    ) -> "NoiseCropDataset":
        print(csv_path)
        df = load_manifest(csv_path, check_files_exist=check_files_exist,
                           check_noise_paths=False)
        return cls(df, rgb_transform=rgb_transform, split=split)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        row   = self.df.iloc[idx]
        label = int(row["label"])

        # RGB crop
        img = Image.open(str(row["face_crop_path"])).convert("RGB")
        if self.rgb_transform is not None:
            img = self.rgb_transform(img)

        # Noise crop: [1, 299, 299] float32, saved by precompute.py
        noise = torch.load(str(row["noise_crop_path"]), map_location="cpu")
        if not isinstance(noise, torch.Tensor):
            raise TypeError(
                f"noise_crop_path {row['noise_crop_path']} did not load a tensor"
            )
        if noise.dim() == 2:
            noise = noise.unsqueeze(0)  # H×W → 1×H×W
        noise = noise.float()

        video_id = str(row["video_id"])
        return img, noise, label, video_id
