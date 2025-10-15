# main.py
import argparse
import subprocess
import sys
from pathlib import Path

from Models.naive_bayes import train_and_evaluate as run_nb
from Models.logistic_regression import train_and_evaluate as run_lr
from Models.classification_tree import train_and_evaluate as run_dt
from Models.random_forest import train_and_evaluate as run_rf
from Models.gradient_boosting import train_and_evaluate as run_gb

from Analysis.run_all import combine_results
from Models.io_utils import ensure_output_dir


# -------------------- Preprocessing --------------------
def run_preprocess_if_needed(csv_path: Path):
    """Automatically preprocess only the 'negative_polarity' subset if needed."""
    if csv_path.exists():
        print(f"[OK] Using existing CSV: {csv_path}")
        return

    zip_candidate = Path("Data/op_spam_v1.4.zip")
    if zip_candidate.exists():
        print(f"[BUILD] Found {zip_candidate}, preprocessing only 'negative_polarity' subset...")
        subprocess.check_call([
            sys.executable, "Data/Preprocess.py",
            "--zip", str(zip_candidate),
            "--out_csv", str(csv_path),
            "--subset", "negative"
        ])
    else:
        raise SystemExit("❌ Missing both preprocessed_df.csv and op_spam_v1.4.zip in Data/.")


# -------------------- Main Workflow --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="Data/preprocessed_df.csv", help="Path to preprocessed_df.csv")
    ap.add_argument("--dataset_zip", type=str, default=None, help="If CSV missing, build it from this ZIP")
    ap.add_argument("--output_dir", type=str, default="output", help="Where to store all results")
    args = ap.parse_args()

    csv_path = Path(args.data)
    out = ensure_output_dir(args.output_dir)

    run_preprocess_if_needed(csv_path)

    # -------------------- Train Models --------------------
    print("\n[1/5] Multinomial Naive Bayes…")
    nb_csv = run_nb(csv_path=str(csv_path), output_dir=str(out))
    print(f"→ {nb_csv}")

    print("\n[2/5] Logistic Regression (L1)…")
    lr_csv = run_lr(csv_path=str(csv_path), output_dir=str(out))
    print(f"→ {lr_csv}")

    print("\n[3/5] Classification Tree…")
    dt_csv = run_dt(csv_path=str(csv_path), output_dir=str(out))
    print(f"→ {dt_csv}")

    print("\n[4/5] Random Forest…")
    rf_csv = run_rf(csv_path=str(csv_path), output_dir=str(out))
    print(f"→ {rf_csv}")

    print("\n[5/5] Gradient Boosting…")
    gb_csv = run_gb(csv_path=str(csv_path), output_dir=str(out))
    print(f"→ {gb_csv}")

    # -------------------- Leaderboard --------------------
    print("\n[Analysis] Building leaderboard…")
    leader = combine_results(output_dir=str(out))
    if leader:
        print(f"→ Leaderboard saved at {leader}")
    else:
        print("⚠️ No model result files found — skipping leaderboard.")

    # -------------------- Model Comparison --------------------
    print("\n[6/6] Generating model comparison graphs...")

    # Locate compare_all_models.py (works if in Data/ or Analysis/)
    compare_script = Path("Analysis/compare_all_models.py")
    if not compare_script.exists():
        compare_script = Path("Data/compare_all_models.py")

    if compare_script.exists():
        print(f"[RUN] {compare_script}")
        subprocess.run([sys.executable, str(compare_script)])
    else:
        print("⚠️ compare_all_models.py not found in Analysis/ or Data/.")

    print("\n✅ All analyses complete. Check the 'output/graphs/' folder for results.")


if __name__ == "__main__":
    main()