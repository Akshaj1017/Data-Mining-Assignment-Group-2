# Models/naive_bayes.py
# ===== Multinomial Naive Bayes for Opinion Spam (Project Version) =====

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    f1_score, make_scorer, confusion_matrix
)
from joblib import dump

from .io_utils import ensure_output_dir

# ---------------- Constants ----------------
POS_LABEL = "DECEPTIVE"
NEG_LABEL = "TRUTHFUL"
RANDOM_STATE = 42


# ---------------- Helper Functions ----------------
def _vectorizers():
    """Return [(tag, vec-pipeline)] for unigrams and uni+bigrams."""
    variants = []
    for ngram in [(1, 1), (1, 2)]:
        tag = "uni" if ngram == (1, 1) else "uni+bi"
        vec = Pipeline([
            ("cv", CountVectorizer(ngram_range=ngram, min_df=2, max_features=20000)),
            ("tfidf", TfidfTransformer(use_idf=True))
        ])
        variants.append((tag, vec))
    return variants


def _make_nb_pipeline(vec, use_topk: bool = False):
    """Naive Bayes pipeline with/without SelectKBest(chi2)."""
    steps = [("vec", vec)]
    if use_topk:
        steps.append(("select", SelectKBest(score_func=chi2, k=5000)))
    steps.append(("clf", MultinomialNB()))
    return Pipeline(steps)


def _top_terms_nb(pipe, pos_label=POS_LABEL, k=10):
    """Extract top-k terms for DECEPTIVE and TRUTHFUL via log-prob difference."""
    vec = pipe.named_steps["vec"]
    cv = vec.named_steps["cv"]
    vocab = cv.get_feature_names_out()
    clf = pipe.named_steps["clf"]

    classes = clf.classes_
    fake_idx = int(np.where(classes == pos_label)[0][0])
    true_idx = 1 - fake_idx

    diff = clf.feature_log_prob_[fake_idx] - clf.feature_log_prob_[true_idx]
    top_fake_idx = np.argsort(diff)[-k:][::-1]
    top_true_idx = np.argsort(diff)[:k]

    return vocab[top_fake_idx].tolist(), vocab[top_true_idx].tolist()


# ---------------- Train & Evaluate ----------------
def train_and_evaluate(csv_path: str = "Data/preprocessed_df.csv", output_dir: str = "output") -> Path:
    """
    Project entrypoint:
      - 5-fold stratified split (fold 5 = test)
      - 4 configs: (uni, uni+bi) × (with/without top-k)
      - Saves:
          nb_results.csv
          nb.confusion_matrix.csv
          nb.top_terms.txt
          nb_best_pipeline.joblib
    """
    out = ensure_output_dir(output_dir)

    df = pd.read_csv(csv_path)
    assert {"Review", "Label"}.issubset(df.columns), "CSV must contain 'Review' and 'Label'."
    X = df["Review"].astype(str).values
    y = df["Label"].values

    # Fold split: folds 1–4 train, fold 5 test
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = list(skf.split(X, y))[4]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Hyperparameters
    alphas = [0.1, 0.5, 1.0]
    k_vals = [500, 1000, 2000]
    SCORER_F1 = make_scorer(f1_score, pos_label=POS_LABEL)

    # Configurations
    rows, best_pipes, test_preds = [], {}, {}
    vec_uni, vec_unibi = _vectorizers()

    configs = []
    configs += [("NB__uni", _make_nb_pipeline(vec_uni[1], use_topk=False),
                 {"clf__alpha": alphas})]
    configs += [("NB__uni_topk", _make_nb_pipeline(vec_uni[1], use_topk=True),
                 {"clf__alpha": alphas, "select__k": k_vals})]
    configs += [("NB__uni+bi", _make_nb_pipeline(vec_unibi[1], use_topk=False),
                 {"clf__alpha": alphas})]
    configs += [("NB__uni+bi_topk", _make_nb_pipeline(vec_unibi[1], use_topk=True),
                 {"clf__alpha": alphas, "select__k": k_vals})]

    # Train & evaluate
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
            "model": name,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "best_params": json.dumps(gs.best_params_)
        })
        best_pipes[name] = best
        test_preds[name] = yhat

    # Save results
    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    results_path = out / "nb_results.csv"
    results.to_csv(results_path, index=False)

    # Confusion matrix for best
    best_name = results.iloc[0]["model"]
    cm = confusion_matrix(y_test, test_preds[best_name], labels=[NEG_LABEL, POS_LABEL])
    pd.DataFrame(cm, index=[NEG_LABEL, POS_LABEL], columns=[NEG_LABEL, POS_LABEL]) \
        .to_csv(out / "nb.confusion_matrix.csv")

    # Top terms
    tfake, ttrue = _top_terms_nb(best_pipes[best_name], k=5)
    (out / "nb.top_terms.txt").write_text(
        "Top terms for Naive Bayes\n"
        f"→ DECEPTIVE: {', '.join(tfake)}\n"
        f"→ TRUTHFUL:  {', '.join(ttrue)}\n",
        encoding="utf-8"
    )

    # Save best pipeline
    dump(best_pipes[best_name], out / "nb_best_pipeline.joblib")

    return results_path
