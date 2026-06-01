# src/report — Table and figure generation

## Contract

**Produces:**
- Thesis tables as tidy CSVs and LaTeX `\input{}`-ready fragments (T1–T10)
- All thesis figures as `matplotlib.Figure` objects saved to disk
- `RESULTS.md` — top-level manifest of every artefact produced in the run

**Consumes:** Evaluation CSVs, prediction CSVs, and robustness/taxonomy CSVs.
**Never** reads model checkpoints or raw images.

**Needs GPU:** No. This module must never import `torch`.

## Key files

| File | Purpose |
|------|---------|
| `tables.py` | One function per table: `make_table_T1(df) -> pd.DataFrame`, etc. |
| `figures/stage1.py` | Stage 1 figures (barplots, bottleneck ladder, robustness curves) |
| `figures/stage2.py` | Stage 2 figures (calibration, regime comparison, shortcut ablation, dL stats) |
| `figures/taxonomy.py` | Failure taxonomy figures (Cramér's V heatmaps, agreement distributions) |
| `results_md.py` | `assemble_results_md(output_root) -> str` — scans output directory for all artefacts |

## Table index

| ID | Thesis section | Content |
|----|---------------|---------|
| T1 | 3.7.2 | Stage 1 video-level on FF++ |
| T2 | 3.7.2 | Stage 1 fixed-fusion variants |
| T3 | 3.7.5 | Stage 1 cross-dataset AUC |
| T4 | 3.7.6 | Stage 1 robustness JPEG + blur |
| T5 | 3.7.8 | Stage 1 Wilcoxon |
| T6 | 3.7.4 | Five-regime cross-dataset |
| T7 | 3.7.4 | Calibration table |
| T8 | 3.7.4 | dL-quartile FF++ per-method |
| T9 | 3.7.4 | dL-quartile Celeb-DF aggregation |
| T10 | 3.7.4 | Pair-sampling distribution |
