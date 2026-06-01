# tests

## Structure

```
tests/
  unit/          Unit tests — fast, no GPU, no filesystem I/O beyond fixtures
  integration/   Integration tests — run the full tiny-corpus pipeline end-to-end
  fixtures/      Static test data and the tiny-corpus fixture builder
```

## Running

```bash
# All tests
pytest

# Unit tests only (fast, <1 min)
pytest tests/unit/

# Integration tests (requires tiny corpus fixture)
python tests/fixtures/build_tiny_corpus.py   # build once
pytest tests/integration/

# With coverage
pytest --cov=src --cov-report=term-missing
```

## Tiny corpus

8 real + 8 fake face crops used by `make tiny` and the integration tests.
Build it once with:
```bash
python tests/fixtures/build_tiny_corpus.py --out tests/fixtures/tiny_corpus/
```
The fixture is pre-built and committed to the repo so CI does not need
the full corpora.

## What is tested

### Unit tests (tests/unit/)

| File | Tests |
|------|-------|
| `test_metrics.py` | ECE on calibrated/miscalibrated predictors; AUC on edge cases; EER monotonicity; Brier score identity |
| `test_threshold_sweep.py` | argmax-MCC on toy dataset; Youden's J formula; bal_acc at uniform threshold |
| `test_bootstrap.py` | CI width decreases with N; video-level vs sample-level resampling |
| `test_aggregation.py` | Wilcoxon Bonferroni p-value scaling for 4 datasets; video grouping key `(dataset, source, video_id)` |
| `test_tsbi_gate.py` | Pair acceptance: gate fires correctly at dL_mean=6, dL_std=3; relaxation fallback selects highest-delta pair |
| `test_shortcuts.py` | N0–N6 and P0–P5 each run on 4 synthetic crops without crashing; N1 self-transfer AUC ≈ 0.5 |
| `test_perturbations.py` | Each perturbation at identity parameter returns exact input; JPEG q=95 output is decodable |
| `test_cramers.py` | Cramér's V = 1.0 on perfect contingency table; = 0.0 on independent table |
| `test_manifest.py` | Schema validation catches missing columns, bad labels, bad splits |

### Integration tests (tests/integration/)

| File | Tests |
|------|-------|
| `test_tiny_pipeline.py` | End-to-end on tiny corpus: crop → noise precompute → SBI → T-SBI → manifests → train (1 epoch, CPU) → eval → report |
