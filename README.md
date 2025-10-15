# Opinion Spam Detection — Ready-to-Run Project

## Layout
```
Data/
  Preprocess.py
  preprocessed_df.csv         # created by Preprocess.py (or place your own)
Models/
  naive_bayes.py
  logistic_reg_l1.py
  classification_tree.py
Analysis/
  run_all.py
main.py
requirements.txt
```

## One command
```bash
python main.py --data Data/preprocessed_df.csv --dataset_zip Data/op_spam_v1.4.zip
```
- If `Data/preprocessed_df.csv` **does not** exist, `main.py` will call `Data/Preprocess.py` to build it.
- If you provide `--dataset_zip`, it will attempt to preprocess from that ZIP (e.g., `op_spam_v1.4.zip`). If both CSV and ZIP are missing, the script will explain what it needs.

## Outputs
All model outputs land in `output/` (created at runtime):
- `nb_results.csv`, `lr_l1_results.csv`, `dt_results.csv` (metrics)
- Confusion matrices (`*.confusion_matrix.csv`)
- Top terms/features (`*.top_terms.txt` where applicable)
- Trained pipelines (`*.joblib`)
- A combined leaderboard at `output/summary_leaderboard.csv`

## Notes
- Assumes two labels: `DECEPTIVE` (positive class) and `TRUTHFUL`.
- Uses a **fixed 5-fold split** and tests on the 5th fold to be faithful to the original scripts.
- You can swap in your own `preprocessed_df.csv` with columns `Review`, `Label`.
