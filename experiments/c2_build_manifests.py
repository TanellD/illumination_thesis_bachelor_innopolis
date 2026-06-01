"""
experiments/c2_build_manifests.py
==================================
Entry point for experiments C.2 + C.3 — manifest assembly.

Assembles per-regime training/val manifests from:
  - FF++ real + supervised fakes CSV (from A.2 crop extraction)
  - Classic SBI labels CSV (from c1_generate_sbi.py)
  - T-SBI labels CSV (from c1_generate_tsbi.py)

Also builds the HIGHDL / LOWDL dL-quartile manifests (C.4) when
--tsbi-csv and --out-dir are provided.

CLI:
    thesis-manifests \
        --ff-csv   outputs/crops/ff++/manifest.csv \
        --sbi-csv  outputs/sbi_labels.csv \
        --tsbi-csv outputs/tsbi_labels.csv \
        --out-dir  outputs/manifests

Regime composition (CLAUDE.md §C.3):
    A      : ff_real + ff_supervised_fake
    B_pure : ff_real + sbi_fake
    B_mix  : ff_real + ff_supervised_fake + sbi_fake
    C_pure : ff_real + sbi_fake + tsbi_fake
    C_mix  : ff_real + ff_supervised_fake + sbi_fake + tsbi_fake
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble per-regime training manifests")
    p.add_argument("--ff-csv",   required=True,
                   help="FF++ manifest CSV (A.2 output)")
    p.add_argument("--sbi-csv",  default=None,
                   help="Classic SBI labels CSV (c1_generate_sbi.py output)")
    p.add_argument("--tsbi-csv", default=None,
                   help="T-SBI labels CSV (c1_generate_tsbi.py output)")
    p.add_argument("--out-dir",  required=True,
                   help="Output directory for regime manifests")
    p.add_argument("--splits",   nargs="+", default=["train", "val"],
                   help="Splits to include (default: train val)")
    return p.parse_args()


def _read_ff(ff_csv: str, splits: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (real_df, fake_df) from the FF++ manifest, filtered to splits."""
    df = pd.read_csv(ff_csv)
    if "split" in df.columns:
        df = df[df["split"].isin(splits)]
    real = df[df["label"] == 0].copy()
    fake = df[df["label"] == 1].copy()
    return real, fake


def _read_gen(csv_path: Optional[str], splits: list,
              source_tag: str) -> Optional[pd.DataFrame]:
    """Read a generation CSV (SBI or T-SBI) and tag the source."""
    if csv_path is None or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"].isin(splits)]
    if "source" not in df.columns:
        df["source"] = source_tag
    return df


def _write_regime(regime_name: str, out_dir: str, frames: list[pd.DataFrame],
                  splits: list) -> None:
    """Concatenate frames and write per-split CSVs."""
    combined = pd.concat([f for f in frames if f is not None and len(f) > 0],
                          ignore_index=True)
    for split in splits:
        sub = combined[combined["split"] == split] if "split" in combined.columns \
              else combined
        out_path = Path(out_dir) / f"{regime_name}_{split}.csv"
        sub.to_csv(out_path, index=False)
        logger.info(f"  {regime_name} {split}: {len(sub):,} rows → {out_path}")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_df, ff_fake_df = _read_ff(args.ff_csv, args.splits)
    sbi_df   = _read_gen(args.sbi_csv,  args.splits, "sbi")
    tsbi_df  = _read_gen(args.tsbi_csv, args.splits, "tsbi")

    logger.info(f"FF++ real:      {len(real_df):,} rows")
    logger.info(f"FF++ fake:      {len(ff_fake_df):,} rows")
    if sbi_df is not None:
        logger.info(f"SBI fakes:      {len(sbi_df):,} rows")
    if tsbi_df is not None:
        logger.info(f"T-SBI fakes:    {len(tsbi_df):,} rows")

    _write_regime("A",      str(out_dir), [real_df, ff_fake_df],          args.splits)
    _write_regime("B_pure", str(out_dir), [real_df, sbi_df],              args.splits)
    _write_regime("B_mix",  str(out_dir), [real_df, ff_fake_df, sbi_df],  args.splits)
    _write_regime("C_pure", str(out_dir), [real_df, sbi_df, tsbi_df],     args.splits)
    _write_regime("C_mix",  str(out_dir), [real_df, ff_fake_df, sbi_df, tsbi_df], args.splits)

    if tsbi_df is not None and args.tsbi_csv:
        logger.info("Building HIGHDL / LOWDL manifests (C.4)...")
        from src.tsbi.gating import build_highdl_lowdl_manifests
        cpure_train = str(out_dir / "C_pure_train.csv")
        cpure_val   = str(out_dir / "C_pure_val.csv")
        if os.path.exists(cpure_train) and os.path.exists(cpure_val):
            build_highdl_lowdl_manifests(
                tsbi_csv=args.tsbi_csv,
                train_manifest_csv=cpure_train,
                val_manifest_csv=cpure_val,
                out_dir=str(out_dir),
            )
        else:
            logger.warning("C_pure manifests not found — skipping HIGHDL/LOWDL split")

    logger.info(f"Manifests written to {out_dir}")


if __name__ == "__main__":
    main()
