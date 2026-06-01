# src/models — Model architecture definitions

## Contract

**Produces:** PyTorch `nn.Module` subclasses. No training logic, no data loading.

**Consumes:** Nothing from other `src/` modules (models are leaf nodes in the import graph).

**Needs GPU:** No at definition time; yes at forward-pass time.

## Models

| Class | File | Input | Backbone | Stage |
|-------|------|-------|----------|-------|
| `RGBOnlyModel` | `rgb_only.py` | 3×299×299 | Xception (ImageNet) | Stage 1 baseline |
| `ResidualOnlyModel` | `residual_only.py` | 1×299×299 noise | ResNet (custom) | Stage 1 — InstanceNorm intentional (see KNOWN_QUIRKS #1) |
| `LateFusionModel` | `late_fusion.py` | 3×299×299 + 1×299×299 | Xception + ResNet | Stage 1 fusion |
| `StatNoiseFusionModel` | `statnoise_fusion.py` | 3×299×299 + 13 scalars | Xception + MLP | Stage 1 B.4a |
| `ResAwareFusionModel` | `resaware_fusion.py` | 3×H×W + 1×H×W (native res) | Xception + ResNet (BatchNorm) | Stage 1 B.4b |
| `EfficientNetB4Model` | `efficientnet.py` | 3×380×380 | EfficientNet-B4 (timm) | Stage 2 |
| `NoiseprintPlusPlus` | `noiseprintpp.py` | 3×H×W | 17-layer FCN (frozen) | Noise extractor |

## Critical constraints

- Do NOT replace `InstanceNorm2d` with `BatchNorm2d` in `ResidualOnlyModel` — see KNOWN_QUIRKS.md #1.
- Do NOT replace `BatchNorm2d` with `InstanceNorm2d` in `LateFusionModel` — see KNOWN_QUIRKS.md #2.
- Do NOT change backbone types (Xception for Stage 1, EfficientNet-B4 for Stage 2).
- Do NOT change input resolutions (299×299 Stage 1, 380×380 Stage 2).
