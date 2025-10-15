# Analysis/compare_all_models.py
# Compare acc/prec/recall/F1 for ALL model configurations (15 rows) and plot bar charts.

from __future__ import annotations
import json, ast
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("output")
PLOT_DIR = OUTPUT_DIR / "graphs"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILES = [
    "dt_results.csv",      # multiple alpha rows (DT__...__alpha=...)
    "gb_results.csv",      # 2 rows (GB__uni, GB__uni+bi)
    "nb_results.csv",      # 4 rows (NB__uni, NB__uni_topk, NB__uni+bi, NB__uni+bi_topk)
    "rf_results.csv",      # 2 rows (RF__uni, RF__uni+bi)
    "lr_l1_results.csv",   # 2 rows (LR_L1__uni, LR_L1__uni+bi)
]

def _read_results(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    # Ensure numeric metrics
    for col in ["acc", "prec", "rec", "f1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional family inference if not already present
    if "family" not in df.columns:
        fam = (
            "DT" if path.name.startswith("dt_") else
            "GB" if path.name.startswith("gb_") else
            "NB" if path.name.startswith("nb_") else
            "RF" if path.name.startswith("rf_") else
            "LR_L1" if path.name.startswith("lr_l1_") else
            "UNKNOWN"
        )
        df["family"] = fam

    return df

def _concat_all() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for fname in RESULT_FILES:
        df = _read_results(OUTPUT_DIR / fname)
        if df is not None:
            frames.append(df)
        else:
            print(f"[WARN] Missing or empty: {fname}")
    if not frames:
        raise FileNotFoundError("No result CSVs found in 'output/'. Run your models first.")
    all_df = pd.concat(frames, ignore_index=True)

    # Normalize 'model' to string (some pandas types can be non-string)
    all_df["model"] = all_df["model"].astype(str)

    # Sort by F1 (desc)
    all_df = all_df.sort_values("f1", ascending=False).reset_index(drop=True)
    # Save a combined CSV for convenience
    combined_path = OUTPUT_DIR / "all_models_results.csv"
    all_df.to_csv(combined_path, index=False)
    print(f"[SAVED] Combined results → {combined_path}")
    return all_df

def _bar_plot(df: pd.DataFrame, metric: str, out_path: Path, title: str):
    """One figure per metric; x-axis is model (15 bars)."""
    x_labels = df["model"].tolist()
    y_vals = df[metric].tolist()

    plt.figure(figsize=(14, 7))
    idx = np.arange(len(x_labels))
    plt.bar(idx, y_vals)
    plt.xticks(idx, x_labels, rotation=45, ha="right")
    plt.ylabel(metric.upper())
    plt.ylim(0, 1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    df = _concat_all()

    # Four bar charts (one per metric) over ALL rows
    for metric in ["acc", "prec", "rec", "f1"]:
        out = PLOT_DIR / f"all_models__{metric}_bars.png"
        _bar_plot(
            df=df,
            metric=metric,
            out_path=out,
            title=f"All Models — {metric.upper()} (sorted by F1 overall)"
        )
        print(f"[SAVED] {out}")

    print(f"\n✅ Done. Charts saved in {PLOT_DIR.resolve()}")

if __name__ == "__main__":
    main()
