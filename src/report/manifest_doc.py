"""
src/report/manifest_doc.py
==========================
RESULTS.md assembler (G.3).

`write_results_md(out_dir, run_info)` produces a top-level RESULTS.md
summarising which experiments ran, their seeds, git SHA, and pointing to
every CSV and figure produced.

The file is written from the structured run_info dict returned by
experiments/g_report.py after a full run.
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def write_results_md(
    out_dir: str,
    run_info: Dict[str, Any],
) -> str:
    """Write RESULTS.md to out_dir and return the path.

    Parameters
    ----------
    out_dir  : directory where RESULTS.md is written (usually the run root)
    run_info : dict with keys:
        config_hash (str)
        git_sha (str)
        timestamp (str ISO-8601)
        python_version (str)
        torch_version (str)
        hostname (str)
        gpu (str)
        tables (dict: table_id -> csv_path)
        figures (dict: figure_id -> png_path)
        experiments (list of dicts: name, seeds, elapsed_s, output_dir)
    """
    lines: List[str] = []

    def _h(level: int, text: str) -> None:
        lines.append(f"{'#' * level} {text}")

    def _p(text: str) -> None:
        lines.append(text)

    def _blank() -> None:
        lines.append("")

    _h(1, "RESULTS")
    _blank()
    _p(f"Generated: {run_info.get('timestamp', datetime.datetime.utcnow().isoformat())}")
    _p(f"Git SHA:   `{run_info.get('git_sha', 'unknown')}`")
    _p(f"Config:    `{run_info.get('config_hash', 'unknown')}`")
    _p(f"Host:      {run_info.get('hostname', 'unknown')}  "
       f"GPU: {run_info.get('gpu', 'cpu')}")
    _p(f"Python:    {run_info.get('python_version', '?')}  "
       f"PyTorch: {run_info.get('torch_version', '?')}")
    _blank()

    # ── Experiments ─────────────────────────────────────────────────────────
    experiments = run_info.get("experiments", [])
    if experiments:
        _h(2, "Experiments")
        _blank()
        _p("| Experiment | Seeds | Elapsed | Output directory |")
        _p("|------------|-------|---------|------------------|")
        for exp in experiments:
            seeds_str = ", ".join(str(s) for s in exp.get("seeds", []))
            _p(f"| {exp.get('name', '?')} "
               f"| {seeds_str or '—'} "
               f"| {exp.get('elapsed_s', '?')}s "
               f"| `{exp.get('output_dir', '?')}` |")
        _blank()

    # ── Tables ──────────────────────────────────────────────────────────────
    tables = run_info.get("tables", {})
    if tables:
        _h(2, "Tables (T1–T10)")
        _blank()
        _p("| Table | CSV | LaTeX |")
        _p("|-------|-----|-------|")
        for tid in sorted(tables.keys()):
            csv_path = tables[tid].get("csv", "pending")
            tex_path = tables[tid].get("tex", "pending")
            status   = "✓" if os.path.exists(csv_path) else "⏳"
            _p(f"| {tid} {status} | `{csv_path}` | `{tex_path}` |")
        _blank()

    # ── Figures ──────────────────────────────────────────────────────────────
    figures = run_info.get("figures", {})
    if figures:
        _h(2, "Figures")
        _blank()
        _p("| Figure | Path | Status |")
        _p("|--------|------|--------|")
        for fid, fpath in sorted(figures.items()):
            status = "✓" if os.path.exists(str(fpath)) else "⏳"
            _p(f"| {fid} | `{fpath}` | {status} |")
        _blank()

    # ── Evaluation CSVs ───────────────────────────────────────────────────────
    eval_csvs = run_info.get("eval_csvs", {})
    if eval_csvs:
        _h(2, "Evaluation CSVs")
        _blank()
        for key, path in sorted(eval_csvs.items()):
            status = "✓" if os.path.exists(str(path)) else "⏳"
            _p(f"- [{key}]({path}) {status}")
        _blank()

    # ── Reproduction commands ─────────────────────────────────────────────────
    _h(2, "Reproduction")
    _blank()
    _p("To reproduce from scratch with the full corpora:")
    _p("```bash")
    _p("# 1. Data preparation")
    _p("make full-data")
    _p("")
    _p("# 2. Stage 1 training + evaluation")
    _p("make stage1")
    _p("")
    _p("# 3. Stage 2 T-SBI generation + training + evaluation")
    _p("make stage2")
    _p("")
    _p("# 4. Robustness + taxonomy")
    _p("make robustness taxonomy")
    _p("")
    _p("# 5. Report")
    _p("make report")
    _p("```")
    _blank()
    _p("To reproduce only the report layer from cached CSVs:")
    _p("```bash")
    _p("thesis-g-report --config configs/eval/report.yaml")
    _p("```")
    _blank()

    # ── Notes ─────────────────────────────────────────────────────────────────
    notes = run_info.get("notes", [])
    if notes:
        _h(2, "Notes")
        _blank()
        for note in notes:
            _p(f"- {note}")
        _blank()

    md = "\n".join(lines)
    out_path = os.path.join(out_dir, "RESULTS.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    return out_path
