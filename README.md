# Opinion Spam Detection 

## 🚀 How to Run the Project

From your project root directory (for example `SPAM_DETECTION_PROJECT`), open a terminal and run:

```bash
python main.py
```

### 🔍 What Happens When You Run It

1. Checks if `Data/preprocessed_df.csv` exists.

   * If missing, it automatically runs preprocessing using `Data/op_spam_v1.4.zip` (negative polarity subset).
2. Trains and evaluates the following models:

   * Multinomial Naive Bayes
   * Logistic Regression (L1 regularization)
   * Classification Tree
   * Random Forest
   * Gradient Boosting
3. Saves all model outputs in the `output/` folder.
4. Generates a combined **leaderboard** and model performance comparison graphs.

Example console output:

```
[1/5] Multinomial Naive Bayes…
→ output/naive_bayes_results.csv

[Analysis] Building leaderboard…
→ Leaderboard saved at output/leaderboard.csv

✅ All analyses complete. Check the 'output/graphs/' folder for results.
```

---

## 📊 Key Scripts and Their Roles

| Script                             | Purpose                                                                  | Output                                         |
| ---------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------- |
| **main.py**                        | Main entry point — runs preprocessing, model training, and analysis.     | Triggers all models and produces final outputs |
| **Data/Preprocess.py**             | Cleans and prepares data from `op_spam_v1.4.zip`.                        | `Data/preprocessed_df.csv`                     |
| **Models/**                        | Contains scripts for each classifier (`train_and_evaluate()` functions). | Individual model results (`*.csv`)             |
| **Analysis/run_all.py**            | Combines results from all models into a single leaderboard.              | `output/leaderboard.csv`                       |
| **Analysis/compare_all_models.py** | Generates performance comparison graphs.                                 | Graphs in `output/graphs/`                     |

> The **performance metrics** (Accuracy, Precision, Recall, F1) reported in your project’s results section are produced by the model scripts executed inside `main.py`.

---

## 📦 Dependencies

Install the required libraries using:

```bash
pip install -r requirements.txt
```

### Core Packages

* `numpy`
* `pandas`
* `scikit-learn`
* `nltk`
* `spacy`
* `joblib`
* `matplotlib`

### Optional (Recommended)

To enable lemmatization and advanced text cleaning:

```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords wordnet
```

---

## 📁 Outputs

All generated results are saved in the `output/` directory.

| File / Folder            | Description                                                |
| ------------------------ | ---------------------------------------------------------- |
| `output/*.csv`           | Accuracy, precision, recall, and F1 metrics for each model |
| `output/leaderboard.csv` | Combined summary across all models                         |
| `output/graphs/`         | Visualization of model performance comparisons             |

---

## ✅ Example Workflow

```bash
# Full pipeline (preprocess + train + analyze)
python main.py

# Optional: rerun only the analysis step later
python -m Analysis.run_all
```

Expected completion message:

```
✅ All analyses complete. Check the 'output/graphs/' folder for results.
```

---

