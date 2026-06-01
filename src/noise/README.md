# src/noise — Noiseprint++ extraction and noise diagnostics

## Contract

**Produces:**
- Precomputed noise-map `.pt` files (one per face crop) + updated manifest CSV with `noise_crop_path` column
- Scalar noise statistics (`mean_abs`, `std`, `energy`, `max_abs`, `kurtosis`, `n_pixels`, + 7 extended stats)

**Consumes:** Face-crop manifests from `src/data/`. Noiseprint++ weights from `paths.yaml`.

**Needs GPU:** Yes for `precompute.py` (Noiseprint++ inference); No for `stats.py`.

## Key files

| File | Purpose |
|------|---------|
| `precompute.py` | Run Noiseprint++ on full frames, then crop to face bbox. **Extract-then-crop order is load-bearing** — see KNOWN_QUIRKS.md #3. |
| `stats.py` | Pure numpy scalar feature extraction. Exposes `extract_6()` (used in B.2 bottleneck) and `extract_13()` (used in B.4a StatNoise-Fusion). |

## Critical constraint — extraction order

```
CORRECT:   full_frame → Noiseprint++() → noise_map[H×W] → crop to bbox → noise_crop
WRONG:     crop to bbox → Noiseprint++() on crop → noise_crop
```

The wrong order changes the noise values near crop boundaries because the network's
receptive field sees different context. See KNOWN_QUIRKS.md #3.
