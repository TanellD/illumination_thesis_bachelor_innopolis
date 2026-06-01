# Illumination & Imaging Physics Invariants — Photometric Forensics

Reproducible pipeline for the bachelor thesis:
**"Illumination & Imaging Physics Invariants — Photometric Forensics"**

---

## Pipeline DAG

```
┌─────────────────────────────────────────────────────────────────────────┐
│ A. Data preparation (shared)                                            │
│                                                                         │
│  A.1 corpus ingestion (FF++, Celeb-DF, DFDC, DFF)                      │
│       └─► A.2 MTCNN face extraction  ──► crops/ + manifests            │
│               ├─► A.3 small corpus (Stage 1, ~28k)                     │
│               └─► A.3 large corpus (Stage 2, ~77k)                     │
│  A.5 Noiseprint++ noise precomputation  (Stage 1 only)                 │
└─────────────────────────────────────────────────────────────────────────┘
          │                           │
          ▼                           ▼
┌──────────────────────┐   ┌──────────────────────────────────────────────┐
│ B. Stage 1           │   │ C. Stage 2 (T-SBI)                           │
│                      │   │                                               │
│ B.1 3-model ablation │   │ C.1 Classic SBI generation                  │
│     RGB-Only         │   │ C.1 T-SBI generation                        │
│     Residual-Only    │   │ C.2 Pair illumination gating + manifests    │
│     Late-Fusion      │   │ C.3 5-regime training                        │
│ B.2 Bottleneck diag. │   │     A / B_pure / B_mix / C_pure / C_mix     │
│ B.3 Context-crop     │   │ C.4 HIGHDL / LOWDL multi-seed diagnostic    │
│ B.4 Fixed-fusion     │   │ C.5 Shortcut ablation (N0-N6, P0-P5)       │
│     StatNoise-Fusion │   └──────────────────────────────────────────────┘
│     ResAware-Fusion  │                     │
└──────────────────────┘                     │
          │                                  │
          └──────────────┬───────────────────┘
                         ▼
          ┌──────────────────────────────────┐
          │ D-F. Shared evaluation           │
          │                                  │
          │ D. Metrics (AUC, EER, FPR@95,   │
          │    ECE, Brier), video aggregation │
          │    bootstrap CIs, threshold sweep │
          │ E. Robustness grid               │
          │    JPEG / blur / gamma / resize  │
          │    denoise+sharpen               │
          │ F. Failure taxonomy              │
          │    9 attributes, chi-square,     │
          │    Cramers V, Jaccard agreement  │
          └──────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────────┐
          │ G. Report                        │
          │                                  │
          │ G.1 Tables T1-T10 (CSV + LaTeX)  │
          │ G.2 Figures (reliability, robust, │
          │     bar plots)                   │
          │ G.3 RESULTS.md                   │
          └──────────────────────────────────┘
```

---

## Chapter-experiment map

| Thesis chapter | Experiments | CLI commands |
|---|---|---|
| §3 Data | A.1-A.5 | `make crop`, `make noise-pre` |
| §4 Stage 1 (noise channel) | B.1-B.4 | `make b1 b2 b3 b4` |
| §5 Stage 2 (T-SBI) | C.1-C.5 | `make sbi tsbi manifests train` |
| §3.7 Evaluation | D.1-D.5, E, F | `make eval robustness taxonomy` |
| §4-5 tables/figures | G.1-G.3 | `make report` |

---

## Quickstart (tiny corpus, < 5 min on CPU)

```bash
# 1. Install
git clone <repo>
cd thesis_refactoring
pip install -e ".[dev]"

# 2. Build the tiny-corpus fixture (8 real + 8 fake synthetic crops)
python tests/fixtures/build_tiny_corpus.py \
    --out tests/fixtures/tiny_corpus/

# 3. Run the unit tests (all blocks 1-6)
make test
# Expected: 241 passed

# 4. Run the tiny end-to-end smoke test
make tiny
# Produces outputs/tiny/RESULTS.md in < 5 min on CPU
```

---

## Full reproduction

Set `paths.yaml` to point at your corpora (or use environment variables):

```bash
export THESIS_FFPP_ROOT=/data/faceforensics
export THESIS_CELEBDF_ROOT=/data/celebdf2
export THESIS_DFDC_ROOT=/data/dfdc
export THESIS_DFF_ROOT=/data/deepfakeface
export THESIS_OUTPUT_ROOT=/outputs
export THESIS_NOISEPRINT_WEIGHTS=artifacts/models_weights/noiseprintplusplus_weights.pth
```

Then run all experiments in dependency order:

```bash
# Step 1: Face extraction (~6 h GPU)
make crop

# Step 2: Noise precomputation (Stage 1 only, ~8 h GPU)
make noise-pre

# Step 3: SBI + T-SBI generation (~8 h GPU combined)
make sbi tsbi

# Step 4: Manifest assembly (< 5 min CPU)
make manifests

# Step 5: Stage 1 training (~70 h GPU total, can run b1-b4 in parallel)
make b1 b2 b3 b4

# Step 6: Stage 2 training (~120 h GPU total)
make train

# Step 7: Evaluation, all model x dataset pairs (~6 h GPU)
make eval

# Step 8: Robustness grid (~20 h GPU)
make robustness

# Step 9: Failure taxonomy (< 2 h CPU)
make taxonomy

# Step 10: Report (< 10 min CPU — no GPU needed)
make report
```

Or run everything at once (~240 GPU-hours total):

```bash
make full
```

Preview the DAG without executing anything:

```bash
make dry-run
```

---

## Reproducing a single experiment with a tweaked parameter

Edit one YAML file, re-run that experiment; everything downstream rebuilds.

```bash
# Example: re-run Stage 1 B.1 with a different seed set
# Edit configs/stage1/b1_ablation.yaml -> seeds: [42, 123, 456, 789, 99999]
thesis-b1 --config configs/stage1/b1_ablation.yaml --force
```

---

## Reproducing every thesis table and figure

Tables and figures are deterministic functions of evaluation CSVs.
The report phase never needs a GPU.

```bash
# From cached CSVs in outputs/predictions/:
thesis-g-report --config configs/eval/report.yaml
# Writes:
#   outputs/g_report/<hash>/tables_csv/T1.csv ... T10.csv
#   outputs/g_report/<hash>/tables_tex/T1.tex ... T10.tex
#   outputs/g_report/<hash>/figures/reliability_diagrams.png
#   outputs/g_report/<hash>/RESULTS.md
```

Tables in thesis order:

| Table | Content | Source function |
|---|---|---|
| T1 | Stage 1 FF++ video AUC | `build_T1` |
| T2 | Fixed-fusion 5-seed mean+std | `build_T2` |
| T3 | Cross-dataset AUC | `build_T3` |
| T4 | JPEG+blur robustness | `build_T4` |
| T5 | Wilcoxon signed-rank | `build_T5` |
| T6 | Five-regime AUC | `build_T6` |
| T7 | ECE / Brier calibration | `build_T7` |
| T8 | dL-quartile per-method | `build_T8` |
| T9 | dL-quartile aggregation | `build_T9` |
| T10 | T-SBI pair statistics | `build_T10` |

---

## Module guide

| Module | Produces | Consumes | GPU? |
|---|---|---|---|
| `src/data/` | Face crops, manifests | Raw video frames | Yes (MTCNN) |
| `src/models/` | nn.Module definitions | — | At forward pass |
| `src/noise/` | Noise .pt files, bottleneck diagnostics | Face crops, NoiseprintPlusPlus weights | Yes |
| `src/train/` | Trained checkpoints | Manifests, models | Yes |
| `src/tsbi/` | T-SBI JPEG crops, illumination stats | Real video frames | No |
| `src/eval/` | Metric dicts, threshold sweeps, CIs | Prediction CSVs | No |
| `src/robustness/` | Long-form robustness CSV | Checkpoints, manifests | Yes (inference) |
| `src/taxonomy/` | Attribute Parquet, contingency CSVs | Face crops, prediction CSVs | No |
| `src/report/` | LaTeX tables, PNG figures, RESULTS.md | Evaluation CSVs, DiskCache | No |

Per-module READMEs: `src/<module>/README.md`

---

## Critical constraints (do not change)

These are load-bearing science decisions that reproduce the reported results:

- `ResidualOnlyModel.noise_norm` **must be InstanceNorm2d** (not BatchNorm2d).
  Swapping it erases the bottleneck diagnostic finding. See `KNOWN_QUIRKS.md` #1.
- `LateFusionModel.noise_norm` **must be BatchNorm2d** (not InstanceNorm2d).
  See `KNOWN_QUIRKS.md` #2.
- Noiseprint++ **must be run on the full video frame first**, then the noise
  map is cropped. Reversing this changes SNR values. See `KNOWN_QUIRKS.md` #3.
- T-SBI illumination gate: `dL_mean >= 6.0 OR dL_std >= 3.0`. See §C.2.
- Video aggregation key: `(dataset, source, video_id)` — never bare `video_id`.
  Mixing SBI/T-SBI fakes with their source reals inflates AUC. See `KNOWN_QUIRKS.md` #12.

---

## Development

```bash
make test        # unit tests only (fast, CPU, ~10 s)
make test-all    # unit + integration smoke test (requires tiny corpus)
make lint        # ruff + mypy
make format      # ruff format
```
