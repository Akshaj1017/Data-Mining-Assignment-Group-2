# Analysis/compare_all_models.py
# Compare acc/prec/recall/F1 for ALL model configurations (15 rows)
# - Four bar charts (one per metric)
# - Four pairwise heatmaps (models x models) showing |metric_i - metric_j|

from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # for heatmaps

OUTPUT_DIR = Path("output")
PLOT_DIR = OUTPUT_DIR / "graphs"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILES = [
    "dt_results.csv",
    "gb_results.csv",
    "nb_results.csv",
    "rf_results.csv",
    "lr_l1_results.csv",
]

# ---------------------- Helpers ----------------------
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

    # Family (optional; not needed for plots)
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
    all_df["model"] = all_df["model"].astype(str)
    # Sort globally by F1 for a consistent order across all plots
    all_df = all_df.sort_values("f1", ascending=False).reset_index(drop=True)

    combined_path = OUTPUT_DIR / "all_models_results.csv"
    all_df.to_csv(combined_path, index=False)
    print(f"[SAVED] Combined results → {combined_path}")
    return all_df


# ---------------------- Plot Functions ----------------------
def _bar_plot(df: pd.DataFrame, metric: str, out_path: Path, title: str):
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


def _pairwise_heatmap(df: pd.DataFrame, metric: str):
    """
    Build a symmetric matrix where entry (i,j) = |metric_i - metric_j|.
    Rows/cols are model names (full strings). Diagonal = 0.
    """
    df_ord = df.copy()  # already sorted by F1 in _concat_all
    names = df_ord["model"].tolist()
    vals = df_ord[metric].to_numpy(dtype=float)

    # Compute pairwise absolute differences
    diff_mat = np.abs(vals[:, None] - vals[None, :])

    # Create DataFrame with names on both axes
    heat_df = pd.DataFrame(diff_mat, index=names, columns=names)

    # Plot
    plt.figure(figsize=(max(10, len(names)*0.6), max(8, len(names)*0.6)))
    sns.heatmap(
        heat_df,
        cmap="YlGnBu",
        square=True,
        annot=True,
        fmt=".2f",
        cbar=True,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 8},
    )
    plt.title(f"Pairwise |Δ {metric.upper()}| Between Models")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path = PLOT_DIR / f"pairwise_heatmap_{metric}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


# ---------------------- Main ----------------------
def main():
    df = _concat_all()

    # 4 bar charts (one per metric)
    for metric in ["acc", "prec", "rec", "f1"]:
        out = PLOT_DIR / f"all_models__{metric}_bars.png"
        _bar_plot(
            df=df,
            metric=metric,
            out_path=out,
            title=f"All Models — {metric.upper()} (sorted by F1 overall)"
        )
        print(f"[SAVED] {out}")

    # 4 pairwise heatmaps (models x models) per metric
    for metric in ["acc", "prec", "rec", "f1"]:
        _pairwise_heatmap(df, metric)

    print(f"\n✅ Done. Bar charts + pairwise heatmaps saved in {PLOT_DIR.resolve()}")


if __name__ == "__main__":
    main()
