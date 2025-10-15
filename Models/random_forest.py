# Models/random_forest.py
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from joblib import dump

from .io_utils import ensure_output_dir

# Constants (kept same concept as your script)
POS_LABEL = "DECEPTIVE"
NEG_LABEL = "TRUTHFUL"
RANDOM_STATE = 42


def _rf_oob_gridsearch(X_train, y_train, n_estimators_values, max_features_values):
    """
    Manual OOB-based grid search for RandomForestClassifier — same concept as your code.
    Returns best_params (dict) and best OOB score.
    """
    best_score = -1.0
    best_params = None

    for n_estimators in n_estimators_values:
        for max_features in max_features_values:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features,
                random_state=RANDOM_STATE,
                oob_score=True,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            oob_score = getattr(rf, "oob_score_", np.nan)
            if oob_score > best_score:
                best_score = oob_score
                best_params = {"n_estimators": n_estimators, "max_features": max_features}

    return best_params, float(best_score)


def _top_terms_from_importance(model: RandomForestClassifier, vectorizer: TfidfVectorizer, k: int = 5):
    """
    Keep your original idea: use overall feature importances to list top features.
    We also mirror your 'DECEPTIVE'/'TRUTHFUL' print by taking top-k and bottom-k.
    """
    importances = model.feature_importances_
    names = vectorizer.get_feature_names_out()

    top_idx = np.argsort(importances)[-k:][::-1]
    low_idx = np.argsort(importances)[:k]

    top_terms = [(names[i], float(importances[i])) for i in top_idx]
    low_terms = [(names[i], float(importances[i])) for i in low_idx]
    return top_terms, low_terms


def _vectorize(train_text, test_text, ngram_range):
    vec = TfidfVectorizer(max_features=20000, min_df=2, ngram_range=ngram_range)
    X_tr = vec.fit_transform(train_text)
    X_te = vec.transform(test_text)
    return vec, X_tr, X_te


def train_and_evaluate(csv_path: str = "Data/preprocessed_df.csv", output_dir: str = "output") -> Path:
    """
    Project-style entrypoint. Runs your Random Forest procedure for:
      - unigrams
      - uni+bigrams
    Uses 5-fold split and evaluates on the 5th fold (same as the other models).
    Saves:
      - rf_results.csv (metrics + best params for each n-gram setting)
      - rf.confusion_matrix.csv (from the best of the two)
      - rf.top_terms.txt (from the best of the two, using feature importances)
      - rf_best_pipeline.joblib (tuple: (best_vectorizer, best_model))
    """
    out = ensure_output_dir(output_dir)

    # Load data
    df = pd.read_csv(csv_path)
    assert {"Review", "Label"}.issubset(df.columns), "CSV must have 'Review' and 'Label' columns."
    X_all = df["Review"].astype(str).values
    y_all = df["Label"].values

    # Fixed split: train on folds 1-4, test on fold 5
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = list(skf.split(X_all, y_all))[4]
    X_train_text, y_train = X_all[train_idx], y_all[train_idx]
    X_test_text, y_test = X_all[test_idx], y_all[test_idx]

    # Hyperparameter grids (same spirit as your code; small tidy ranges)
    n_estimators_values = [40, 80, 120, 160, 200]
    max_features_values = [10, 30, 50, 70, 90]

    rows = []
    best_bundle = None   # (name, vectorizer, model, y_pred)
    best_f1 = -1.0

    # ---- (1) Unigrams ----
    vec_uni, Xtr_uni, Xte_uni = _vectorize(X_train_text, X_test_text, (1, 1))
    params_uni, oob_uni = _rf_oob_gridsearch(Xtr_uni, y_train, n_estimators_values, max_features_values)
    rf_uni = RandomForestClassifier(
        **params_uni, random_state=RANDOM_STATE, oob_score=True, n_jobs=-1
    )
    rf_uni.fit(Xtr_uni, y_train)
    yhat_uni = rf_uni.predict(Xte_uni)

    acc = accuracy_score(y_test, yhat_uni)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, yhat_uni, pos_label=POS_LABEL, average="binary", zero_division=0
    )
    rows.append({
        "model": "RF__uni",
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "best_params": json.dumps(params_uni)
    })
    if f1 > best_f1:
        best_f1 = f1
        best_bundle = ("RF__uni", vec_uni, rf_uni, yhat_uni)

    # ---- (2) Uni+Bigrams ----
    vec_both, Xtr_both, Xte_both = _vectorize(X_train_text, X_test_text, (1, 2))
    params_both, oob_both = _rf_oob_gridsearch(Xtr_both, y_train, n_estimators_values, max_features_values)
    rf_both = RandomForestClassifier(
        **params_both, random_state=RANDOM_STATE, oob_score=True, n_jobs=-1
    )
    rf_both.fit(Xtr_both, y_train)
    yhat_both = rf_both.predict(Xte_both)

    acc = accuracy_score(y_test, yhat_both)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, yhat_both, pos_label=POS_LABEL, average="binary", zero_division=0
    )
    rows.append({
        "model": "RF__uni+bi",
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "best_params": json.dumps(params_both)
    })
    if f1 > best_f1:
        best_f1 = f1
        best_bundle = ("RF__uni+bi", vec_both, rf_both, yhat_both)

    # Save results CSV
    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    results_path = out / "rf_results.csv"
    results.to_csv(results_path, index=False)

    # Confusion matrix for best bundle
    best_name, best_vec, best_model, best_yhat = best_bundle
    cm = confusion_matrix(y_test, best_yhat, labels=[NEG_LABEL, POS_LABEL])
    pd.DataFrame(cm, index=[NEG_LABEL, POS_LABEL], columns=[NEG_LABEL, POS_LABEL]) \
        .to_csv(out / "rf.confusion_matrix.csv")

    # Top terms via feature importances (same concept as your script)
    top_terms, low_terms = _top_terms_from_importance(best_model, best_vec, k=5)
    (out / "rf.top_terms.txt").write_text(
        "Top 5 features for DECEPTIVE (highest importance):\n" +
        "\n".join(f"- '{t}': {w:.6f}" for t, w in top_terms) +
        "\n\nTop 5 features for TRUTHFUL (lowest importance):\n" +
        "\n".join(f"- '{t}': {w:.6f}" for t, w in low_terms),
        encoding="utf-8"
    )

    # Save the trained vectorizer + model together
    dump((best_vec, best_model), out / "rf_best_pipeline.joblib")

    return results_path