# src/taxonomy — Failure taxonomy and attribution

## Contract

**Produces:**
- Per-frame attribute CSV (9 attributes per crop)
- Contingency tables (chi-square, Cramér's V, per-bin failure rates)
- Failure-set Jaccard agreement matrix across models

**Consumes:** Face-crop image files + manifests from `src/data/`. Per-frame prediction CSVs from `src/eval/`.

**Needs GPU:** No. All attribute computation is CPU-only (OpenCV + NumPy).

## Key files

| File | Purpose |
|------|---------|
| `attributes.py` | Compute 9 per-frame attributes: blur (LoG variance), illumination_flatness, illumination_harshness, yaw_magnitude, pitch, eye_state, gaze_deviation, crop_tightness, touches_frame_edge |
| `binning.py` | `quantile_bin(series, n_bins=4, dataset_col)` — bins within each dataset separately to control for dataset-specific distributional shifts |
| `contingency.py` | `chi_square_cramers(failure_status, attribute_bins) -> (chi2, p, cramers_v, per_bin_rates)` |
| `agreement.py` | `failure_jaccard(failure_sets_dict) -> pd.DataFrame` — pairwise Jaccard of failure sets; `count_unique_failures(failure_sets_dict) -> pd.Series` |

## Operating points

Contingency tests run at two thresholds:
1. Default: 0.5
2. `argmax_mcc` threshold from the threshold sweep (computed per model per dataset)
