# Models/gradient_boosting.py
# ===== Gradient Boosting for Opinion Spam (project version) =====
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from joblib import dump

from .io_utils import ensure_output_dir

POS_LABEL = "DECEPTIVE"
NEG_LABEL = "TRUTHFUL"
RANDOM_STATE = 42


# ------------------- Helper Functions -------------------
def _vectorize(train_text, test_text, ngram_range):
    """Vectorize text using TF-IDF."""
    vec = TfidfVectorizer(max_features=20000, min_df=2, ngram_range=ngram_range)
    Xtr = vec.fit_transform(train_text)
    Xte = vec.transform(test_text)
    return vec, Xtr, Xte


def _gridsearch_gb(X_train, y_train, cv_folds, max_depths, learning_rates):
    """Perform GridSearchCV for Gradient Boosting."""
    param_grid = {
        "max_depth": max_depths,
        "learning_rate": learning_rates,
    }
    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def _top_features(model: GradientBoostingClassifier, vectorizer: TfidfVectorizer, k: int = 5):
    """Extract top and bottom k features by importance."""
    importances = model.feature_importances_
    names = vectorizer.get_feature_names_out()

    top_idx = np.argsort(importances)[-k:][::-1]
    low_idx = np.argsort(importances)[:k]

    top_terms = [(names[i], float(importances[i])) for i in top_idx]
    low_terms = [(names[i], float(importances[i])) for i in low_idx]
    return top_terms, low_terms


# ------------------- Main Training Function -------------------
def train_and_evaluate(csv_path: str = "Data/preprocessed_df.csv", output_dir: str = "output") -> Path:
    """
    Project entrypoint:
      - 5-fold split (fold 5 = test)
      - Run Gradient Boosting on unigrams and uni+bigrams
      - Save results and artifacts to output/
    """
    out = ensure_output_dir(output_dir)

    df = pd.read_csv(csv_path)
    assert {"Review", "Label"}.issubset(df.columns), "CSV must contain 'Review' and 'Label'."
    X = df["Review"].astype(str).values
    y = df["Label"].values

    # Create folds (use 5th as test)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = list(skf.split(X, y))[4]
    X_train_text, y_train = X[train_idx], y[train_idx]
    X_test_text, y_test = X[test_idx], y[test_idx]

    # Hyperparameter grid
    max_depths = [2, 4, 6, 8, 10]
    learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    cv_folds = 10

    rows = []
    best_f1 = -1
    best_bundle = None  # (name, vectorizer, model, y_pred)

    # ----------- Unigrams -----------
    vec_uni, Xtr_uni, Xte_uni = _vectorize(X_train_text, X_test_text, (1, 1))
    best_uni, params_uni, score_uni = _gridsearch_gb(Xtr_uni, y_train, cv_folds, max_depths, learning_rates)
    yhat_uni = best_uni.predict(Xte_uni)

    acc = accuracy_score(y_test, yhat_uni)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, yhat_uni, average="binary", pos_label=POS_LABEL, zero_division=0
    )
    rows.append({
        "model": "GB__uni",
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "best_params": json.dumps(params_uni),
        "cv_best_score": score_uni,
    })
    if f1 > best_f1:
        best_f1 = f1
        best_bundle = ("GB__uni", vec_uni, best_uni, yhat_uni)

    # ----------- Uni+Bigrams -----------
    vec_unibi, Xtr_unibi, Xte_unibi = _vectorize(X_train_text, X_test_text, (1, 2))
    best_unibi, params_unibi, score_unibi = _gridsearch_gb(Xtr_unibi, y_train, cv_folds, max_depths, learning_rates)
    yhat_unibi = best_unibi.predict(Xte_unibi)

    acc = accuracy_score(y_test, yhat_unibi)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, yhat_unibi, average="binary", pos_label=POS_LABEL, zero_division=0
    )
    rows.append({
        "model": "GB__uni+bi",
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "best_params": json.dumps(params_unibi),
        "cv_best_score": score_unibi,
    })
    if f1 > best_f1:
        best_f1 = f1
        best_bundle = ("GB__uni+bi", vec_unibi, best_unibi, yhat_unibi)

    # Save results CSV
    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    results_path = out / "gb_results.csv"
    results.to_csv(results_path, index=False)

    # Confusion matrix for best configuration
    best_name, best_vec, best_model, best_yhat = best_bundle
    cm = confusion_matrix(y_test, best_yhat, labels=[NEG_LABEL, POS_LABEL])
    pd.DataFrame(cm, index=[NEG_LABEL, POS_LABEL], columns=[NEG_LABEL, POS_LABEL]) \
        .to_csv(out / "gb.confusion_matrix.csv")

    # Top features (importances)
    top_terms, low_terms = _top_features(best_model, best_vec, k=5)
    (out / "gb.top_terms.txt").write_text(
        "Top 5 features for DECEPTIVE:\n" +
        "\n".join(f"- '{t}': {w:.6f}" for t, w in top_terms) +
        "\n\nTop 5 features for TRUTHFUL:\n" +
        "\n".join(f"- '{t}': {w:.6f}" for t, w in low_terms),
        encoding="utf-8"
    )

    # Save best pipeline
    dump((best_vec, best_model), out / "gb_best_pipeline.joblib")

    return results_path