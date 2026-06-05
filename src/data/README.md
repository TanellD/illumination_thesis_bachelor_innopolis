# src/data — Data preparation layer

## Contract

**Produces:** Face-crop image files + CSV manifests in the format:
```
filename, face_crop_path, label, split, method, x1, y1, x2, y2, confidence, video_id
```

**Consumes:** Raw video/image files from the four source corpora (FF++, Celeb-DF, DFDC, DFF).

**Needs GPU:** Yes (MTCNN runs on CUDA if available; falls back to CPU).

**Does NOT need:** Any model checkpoints, manifests, or outputs from other modules.

## Config

All configurable constants live in `configs/data/<dataset>.yaml`:
- `frames_per_video` — how many frames to sample per video (default: 20 for FF++, 60 for others)
- `crop_size` — output crop resolution in pixels; `null` means no resize (save at bbox size)
- `confidence_threshold` — MTCNN detection confidence cutoff (default: 0.85)
- `padding_ratio` — fractional bbox expansion in each direction (default: 0.3)
- `seed` — random seed for frame sampling and split assignment

## Key files

| File | Purpose |
|------|---------|
| `extract_mtcnn.py` | Single-pass MTCNN extraction; config-driven; works for all four datasets |
| `ffpp_splits.py` | FF++ actor-disjoint split assignment (`parse_filename`, `create_splits`, `assign_split`) |
| `mtcnn_utils.py` | Shared MTCNN init, `crop_with_padding`, confidence filtering |
| `dataset.py` | `FaceCropDataset` and `NoiseCropDataset` — PyTorch Dataset classes used by all training scripts |
| `manifest.py` | `ManifestSchema` dataclass; `validate_manifest(df)` |

## Experiments that use this module

- A.2 face detection + cropping (all four datasets)
- A.5 noise-map precomputation (via `src/noise/precompute.py`, which reads manifests from here)
- All training and evaluation experiments (via `dataset.py`)
