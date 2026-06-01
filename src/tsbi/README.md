# src/tsbi — T-SBI and SBI generators, manifests, shortcut ablations

## Contract

**Produces:**
- SBI crop images + `sbi_labels.csv`
- T-SBI crop images + `tsbi_labels.csv` (includes `dL_mean`, `dL_std`, `illum_relaxed` columns)
- Per-regime training manifests (A, B_pure, B_mix, C_pure, C_mix)
- HIGHDL / LOWDL quartile manifests
- Shortcut ablation AUC results (N0–N6, P0–P5)

**Consumes:** FF++ real face crops + manifests from `src/data/`. Raw FF++ videos (for T-SBI temporal pair selection).

**Needs GPU:** MTCNN in `generate_tsbi.py` uses GPU if available. Manifest assembly and shortcut ablation training are CPU/GPU as configured.

## Key files

| File | Purpose |
|------|---------|
| `generate_sbi.py` | Classic SBI generator. Illumination gate applied. |
| `generate_tsbi.py` | T-SBI generator. Temporal pair selection, illumination gating, five transfer modes. |
| `manifests.py` | Five-regime manifest assembly + HIGHDL/LOWDL quartile split. |
| `shortcuts.py` | N0–N6 T-SBI null variants + P0–P5 SBI null variants. |

## Critical defaults — do not change

| Parameter | Value | Location |
|-----------|-------|----------|
| Illumination gate | `dL_mean >= 6 OR dL_std >= 3` | `configs/stage2/tsbi.yaml` |
| Relaxation fallback | enabled (pick largest available delta) | `configs/stage2/tsbi.yaml` |
| Transfer mode (default) | `histmatch` | `configs/stage2/tsbi.yaml` |
| Pair scale tolerance | 1.3× | `configs/stage2/tsbi.yaml` |
| Pair centre-offset tolerance | 35% | `configs/stage2/tsbi.yaml` |
| Padding ratio | 0.3 | inherited from `configs/data/ffpp.yaml` |
| JPEG quality range | [75, 98] | `configs/stage2/tsbi.yaml` |

## Transfer modes

Five modes available; all must be preserved for ablations (C.5):
`histmatch` (default), `reinhard`, `lowfreq`, `intrinsic`, `gainmap`
