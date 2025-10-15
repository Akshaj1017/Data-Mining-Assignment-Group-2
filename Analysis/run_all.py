import pandas as pd
from pathlib import Path

def combine_results(output_dir="output"):
    out = Path(output_dir)
    pieces = []
    for name in ["nb_results.csv", "lr_l1_results.csv", "dt_results.csv", "gb_results.csv", "rf_results.csv"]:
        p = out / name
        if p.exists():
            df = pd.read_csv(p)
            df["family"] = name.split("_")[0].upper()
            pieces.append(df)
    if not pieces:
        return None
    leaderboard = pd.concat(pieces, ignore_index=True).sort_values("f1", ascending=False)
    leader_path = out / "summary_leaderboard.csv"
    leaderboard.to_csv(leader_path, index=False)
    return leader_path
