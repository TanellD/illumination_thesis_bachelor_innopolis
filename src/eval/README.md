# src/eval — Metric computation and evaluation utilities

## Contract

**Produces:** Metric dicts, threshold-sweep CSVs, bootstrap CI CSVs, aggregated cross-seed tables.

**Consumes:** Per-frame prediction CSVs (score, label, video_id, dataset, source columns).

**Needs GPU:** No. This entire module is CPU-only — a key design requirement so the
analysis phase can run on a laptop without a GPU.

## Key files

| File | Purpose |
|------|---------|
| `metrics.py` | `compute_frame_metrics(scores, labels) -> dict` and `compute_video_metrics(scores, labels, video_ids, strategy) -> dict`. Strategies: `mean`, `max`, `vote`. |
| `threshold_sweep.py` | `sweep_thresholds(scores, labels, n=99) -> pd.DataFrame`. Reports `bal_acc`, `mcc`, `youden_j` at each threshold and the argmax of each. F1 is deliberately excluded. |
| `bootstrap.py` | `bootstrap_auc_ci(scores, labels, video_ids, n_resamples=1000, level=0.95) -> (lo, hi)`. Resamples over videos for cross-dataset sets, over samples for DFF. |
| `aggregation.py` | `aggregate_seeds(results_list) -> (mean_dict, std_dict)` and `wilcoxon_bonferroni(per_sample_errors, datasets) -> p_value_dict`. Grouping key for video-level is `(dataset, source, video_id)` — see KNOWN_QUIRKS.md #12. |
| `cache.py` | `DiskCache` — NPZ-backed per-(model, dataset) inference cache. Enables partial reruns without re-running inference. |

## Metric keys

All output dicts use these exact keys (matching the thesis CSV columns):
`auc`, `eer`, `fpr_at_tpr95`, `ece`, `brier`, `bal_acc`, `mcc`, `youden_j`

ECE uses 15 uniform bins.

## Video-level grouping key

Always group by `(dataset, source, video_id)`, never by `video_id` alone.
Mixing these collapses SBI/T-SBI fakes (whose `source` field is the real
video stem) with genuine real frames, inflating AUC. See KNOWN_QUIRKS.md #12.
