# Models/logistic_regression.py
# ===== Logistic Regression (L1 / Lasso) for Opinion Spam (Project Version) =====

from __future__ import annotations
import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    f1_score, make_scorer, confusion_matrix
)
from joblib import dump

from .io_utils import ensure_output_dir

warnings.filterwarnings("ignore")

# ---------------- Constants ----------------
POS_LABEL = "DECEPTIVE"
NEG_LABEL = "TRUTHFUL"
RAND_SEED = 42
C_GRID = [0.01, 0.1, 1, 10, 100]
MAX_ITER = 5000


# ---------------- Helper Functions ----------------
def _vectorizers():
    """Return vectorizer pipelines for unigrams and uni+bigrams."""
    out = []
    for ngram in [(1, 1), (1, 2)]:
        tag = "uni" if ngram == (1, 1) else "uni+bi"
        vec = Pipeline([
            ("cv", CountVectorizer(ngram_range=ngram, min_df=2, max_features=20000)),
            ("tfidf", TfidfTransformer(use_idf=True)),
        ])
        out.append((tag, vec))
    return out


def _make_lr_l1(vec):
    """Build Logistic Regression (L1/Lasso) pipeline."""
    return Pipeline([
        ("vec", vec),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=MAX_ITER,
            random_state=RAND_SEED
        ))
    ])


def _top_terms_lr(pipe: Pipeline, k: int = 10):
    """Extract top-k features for DECEPTIVE and TRUTHFUL."""
    vec = pipe.named_steps["vec"]
    cv = vec.named_steps["cv"]
    vocab = cv.get_feature_names_out()
    clf = pipe.named_steps["clf"]

    coefs = clf.coef_.ravel()
    top_pos_idx = np.argsort(coefs)[-k:][::-1]  # DECEPTIVE
    top_neg_idx = np.argsort(coefs)[:k]         # TRUTHFUL

    pos_terms = vocab[top_pos_idx].tolist()
    neg_terms = vocab[top_neg_idx].tolist()
    return pos_terms, neg_terms


# ---------------- Main Train & Evaluate ----------------
def train_and_evaluate(csv_path: str = "Data/preprocessed_df.csv", output_dir: str = "output") -> Path:
    """
    Project entrypoint:
      - 5-fold stratified split (fold 5 = test)
      - 2 configs: uni & uni+bi (L1 regularization)
      - Saves:
          lr_l1_results.csv
          lr_l1.confusion_matrix.csv
          lr_l1.top_terms.txt
          lr_l1_best_pipeline.joblib
    """
    out = ensure_output_dir(output_dir)

    # Load data
    df = pd.read_csv(csv_path)
    assert {"Review", "Label"}.issubset(df.columns), "CSV must contain 'Review' and 'Label'."
    X = df["Review"].astype(str).values
    y = df["Label"].values
    assert set(np.unique(y)) == {"DECEPTIVE", "TRUTHFUL"}, f"Unexpected labels: {set(np.unique(y))}"

    # 5-fold split (fold 5 test)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RAND_SEED)
    train_idx, test_idx = list(skf.split(X, y))[4]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Build configs
    SCORER_F1 = make_scorer(f1_score, pos_label=POS_LABEL)
    rows, best_pipes, preds = [], {}, {}

    vec_uni, vec_unibi = _vectorizers()
    configs = []
    configs += [("LR_L1__uni", _make_lr_l1(vec_uni[1]), {"clf__C": C_GRID})]
    configs += [("LR_L1__uni+bi", _make_lr_l1(vec_unibi[1]), {"clf__C": C_GRID})]

    # Train + Evaluate
    for name, pipe, grid in configs:
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            cv=5,
            n_jobs=-1,
            scoring=SCORER_F1,
            refit=True,
            verbose=0
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        yhat = best.predict(X_test)

        acc = accuracy_score(y_test, yhat)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, yhat, average="binary", pos_label=POS_LABEL, zero_division=0
        )
        rows.append({
            "model": name, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "best_params": json.dumps(gs.best_params_)
        })
        best_pipes[name] = best
        preds[name] = yhat

    # Results DataFrame
    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    results_path = out / "lr_l1_results.csv"
    results.to_csv(results_path, index=False)

    # Best config → Confusion Matrix, Top Terms, Pipeline
    best_name = results.iloc[0]["model"]
    best_pipe = best_pipes[best_name]
    best_pred = preds[best_name]

    cm = confusion_matrix(y_test, best_pred, labels=[NEG_LABEL, POS_LABEL])
    pd.DataFrame(cm, index=[NEG_LABEL, POS_LABEL], columns=[NEG_LABEL, POS_LABEL]) \
        .to_csv(out / "lr_l1.confusion_matrix.csv")

    pos_terms, neg_terms = _top_terms_lr(best_pipe, k=10)
    (out / "lr_l1.top_terms.txt").write_text(
        "Top terms for Logistic Regression (L1)\n"
        f"→ DECEPTIVE: {', '.join(pos_terms)}\n"
        f"→ TRUTHFUL:  {', '.join(neg_terms)}\n",
        encoding="utf-8"
    )

    # Save pipeline
    dump(best_pipe, out / "lr_l1_best_pipeline.joblib")

    return results_path