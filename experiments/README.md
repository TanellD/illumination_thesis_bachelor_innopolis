# experiments — Entry-point scripts

Each file here is a thin CLI wrapper that wires together `src/` modules for one
numbered experiment from the thesis. They contain no logic of their own — all
logic lives in `src/`.

## Index

| File | Experiment | GPU | Runtime (full corpus) |
|------|------------|-----|----------------------|
| `a1_ingest.py` | A.1 source corpus ingestion / validation | No | ~5 min |
| `a2_crop.py` | A.2 MTCNN face detection + cropping (all datasets) | Yes | ~6 h |
| `a5_noise_precompute.py` | A.5 Noiseprint++ noise-map precomputation | Yes | ~8 h |
| `b1_ablation.py` | B.1 three-model ablation (5 seeds) | Yes | ~40 h |
| `b2_bottleneck.py` | B.2 seven-level bottleneck diagnostic (5-fold CV) | No | ~1 h |
| `b3_context_crop.py` | B.3 context-crop experiment (1.3× vs 2.7×) | Yes | ~2 h |
| `b4_fixed_fusion.py` | B.4 StatNoise-Fusion + ResAware-Fusion (5 seeds each) | Yes | ~30 h |
| `c1_generate_sbi.py` | C.1 classic SBI generation | Yes (MTCNN) | ~3 h |
| `c1_generate_tsbi.py` | C.1 T-SBI generation | Yes (MTCNN) | ~5 h |
| `c2_build_manifests.py` | C.2 + C.3 regime manifest assembly | No | <5 min |
| `c3_train_regimes.py` | C.3 five-regime training ablation + C.4 multi-seed dL-quartile | Yes | ~120 h |
| `c4_dl_quartile.py` | C.4 HIGHDL/LOWDL manifest build + 3-seed training | Yes | ~40 h |
| `c5_shortcuts.py` | C.5 T-SBI (N0–N6) + SBI (P0–P5) shortcut ablations | Yes (1 epoch) | ~4 h |
| `d_eval.py` | D.1–D.5 evaluation on all (model, dataset) pairs | Yes | ~6 h |
| `e_robustness.py` | E perturbation grid | Yes | ~20 h |
| `f_taxonomy.py` | F.1–F.4 failure taxonomy | No | ~2 h |
| `g_report.py` | G.1–G.3 tables, figures, RESULTS.md | No | ~10 min |

## Usage

All experiments read from `paths.yaml` for corpus locations and write to
`outputs/<experiment>/<config_hash>/`. Every experiment accepts:

```
--config configs/<stage>/<name>.yaml   # required
--force                                # rerun even if outputs exist
--dry-run                              # print what would run, then exit
```

## Dependency order

```
a1 → a2 → a5
           ↓
     b1, b2, b3, b4    (Stage 1; b2/b3 only need a2)
           ↓
     c1 → c2 → c3 → c4 → c5    (Stage 2)
           ↓
     d (eval, shared by both stages)
           ↓
     e (robustness, needs d)
     f (taxonomy, needs d)
           ↓
     g (report, needs d + e + f)
```
