# Models/classification_tree.py
# ========================================
# SINGLE DECISION TREES (two ccp_alpha variants per model)
# - Folds 1–4: train + CV; Fold 5: test only
# - Models: DT__uni, DT__uni+bi
# - For each: fix best hyperparams from CV, then refit with TWO alphas:
#       (best_alpha, contrast_alpha) -> both rows in results table
# - Outputs -> output/dt_results.csv (+ confusion matrices, top terms)
# ========================================

from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from .io_utils import ensure_output_dir

warnings.filterwarnings("ignore", category=UserWarning)

POS_LABEL = "DECEPTIVE"
NEG_LABEL = "TRUTHFUL"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------- utils ----------
def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _make_pipe(ngram_range=(1, 1)) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngram_range,
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            sublinear_tf=True,
            min_df=1,
            max_features=15000
        )),
        ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
        ("tree", DecisionTreeClassifier(random_state=RANDOM_SEED))
    ])


# Slightly wide CV grid (as in your original script)
_PARAM_GRID = {
    "tree__criterion": ["entropy"],
    "tree__max_depth": [None, 40],
    "tree__min_samples_leaf": [1, 5],
    "tree__ccp_alpha": [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
}
_INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


def _fit_cv_and_compare_alphas(
    tag_base: str,
    ngram_range: Tuple[int, int],
    *,
    Xtr,
    ytr,
    Xte,
    yte,
    contrast_alpha: float = 1e-2,
) -> Tuple[pd.DataFrame, int, Dict[str, np.ndarray], Pipeline]:
    """
    1) CV to get best hyperparams (incl. ccp_alpha).
    2) Fix other hyperparams and refit twice: best_alpha + contrast_alpha.
    3) Return metrics rows, disagreement count, ypreds per tag, and the
       pipeline (fitted with best_alpha) for downstream artifacts.
    """
    pipe = _make_pipe(ngram_range)
    gs = GridSearchCV(
        estimator=pipe, param_grid=_PARAM_GRID, scoring="f1_macro",
        cv=_INNER_CV, n_jobs=-1, refit=True, verbose=0
    )
    gs.fit(Xtr, ytr)
    best_params = gs.best_params_.copy()
    best_alpha = best_params["tree__ccp_alpha"]

    fixed = {k: v for k, v in best_params.items() if k != "tree__ccp_alpha"}
    alphas = [best_alpha, contrast_alpha if contrast_alpha != best_alpha else 1e-3]

    rows: List[Dict[str, Any]] = []
    preds: Dict[str, np.ndarray] = {}
    fitted_best = None

    for a in alphas:
        p = _make_pipe(ngram_range)
        p.set_params(**fixed)
        p.set_params(**{"tree__ccp_alpha": a})
        p.fit(Xtr, ytr)
        yhat = p.predict(Xte)

        if a == best_alpha:
            fitted_best = p  # keep the pipeline for artifacts

        acc = accuracy_score(yte, yhat)
        prec, rec, f1, _ = precision_recall_fscore_support(
            yte, yhat, average="binary", pos_label=POS_LABEL, zero_division=0
        )

        tag = f"{tag_base}__alpha={a:.0e}"
        preds[tag] = yhat
        rows.append({
            "model": tag,
            "acc": float(acc),
            "prec": float(prec),
            "rec": float(rec),
            "f1": float(f1),
            "cv_best_alpha": float(best_alpha),
            "cv_best_score_macroF1": float(gs.best_score_),
            "best_params_fixed": json.dumps(fixed | {"tree__ccp_alpha": float(a)})
        })

    # disagreement between the two alpha predictions on test set
    keys = list(preds.keys())
    disagree = int(np.sum(preds[keys[0]] != preds[keys[1]]))
    return pd.DataFrame(rows), disagree, preds, fitted_best


def _save_confusion(cm: np.ndarray, out_path: Path) -> None:
    pd.DataFrame(cm, index=[NEG_LABEL, POS_LABEL], columns=[NEG_LABEL, POS_LABEL]).to_csv(out_path)


def _save_log_odds_top_terms(Xtr_text, ytr, ngram_range, out_dir: Path) -> None:
    """
    Directional log-odds on training set with CountVectorizer
    for the *better* family by CV score.
    """
    cv = CountVectorizer(
        ngram_range=ngram_range, lowercase=True, strip_accents="unicode",
        stop_words="english", min_df=1, max_features=15000
    )
    Xc = cv.fit_transform(Xtr_text)
    vocab = np.array(cv.get_feature_names_out())

    labels = np.unique(ytr)
    if len(labels) != 2:
        return

    if POS_LABEL in labels and NEG_LABEL in labels:
        fake_label, real_label = POS_LABEL, NEG_LABEL
    else:
        fake_label, real_label = labels[0], labels[1]

    mask_fake = (ytr == fake_label)
    mask_real = (ytr == real_label)
    fake_counts = Xc[mask_fake].sum(axis=0).A1
    real_counts = Xc[mask_real].sum(axis=0).A1

    alpha = 0.01
    pf = (fake_counts + alpha) / (fake_counts.sum() + alpha * len(vocab))
    pr = (real_counts + alpha) / (real_counts.sum() + alpha * len(vocab))
    log_odds = np.log(pf / (1 - pf)) - np.log(pr / (1 - pr))

    top5_fake = [vocab[i] for i in np.argsort(-log_odds)[:5]]
    top5_real = [vocab[i] for i in np.argsort(log_odds)[:5]]

    (out_dir / "dt.top_terms.txt").write_text(
        "Top terms (directional log-odds on training set)\n"
        f"→ DECEPTIVE: {', '.join(top5_fake)}\n"
        f"→ TRUTHFUL:  {', '.join(top5_real)}\n",
        encoding="utf-8"
    )


# ---------- public API ----------
def train_and_evaluate(csv_path: str = "Data/preprocessed_df.csv", output_dir: str = "output") -> Path:
    out = ensure_output_dir(output_dir)

    # Load
    df = pd.read_csv(csv_path)
    if not {"Review", "Label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'Review' and 'Label'")
    df = df[["Review", "Label"]].dropna().reset_index(drop=True)
    X_all = df["Review"].astype(str).values
    y_all = df["Label"].astype(str).values

    # Build 5 folds; use 5th as test
    skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold = np.zeros(len(df), dtype=int)
    for i, (_, val_idx) in enumerate(skf_outer.split(X_all, y_all), start=1):
        fold[val_idx] = i
    train_mask = np.isin(fold, [1, 2, 3, 4])
    test_mask = (fold == 5)

    Xtr_text, ytr = X_all[train_mask], y_all[train_mask]
    Xte_text, yte = X_all[test_mask], y_all[test_mask]

    # Run for both n-gram settings
    df_uni, disagree_uni, preds_uni, pipe_uni = _fit_cv_and_compare_alphas(
        "DT__uni", (1, 1), contrast_alpha=1e-2,
        Xtr=Xtr_text, ytr=ytr, Xte=Xte_text, yte=yte
    )
    df_unibi, disagree_unibi, preds_unibi, pipe_unibi = _fit_cv_and_compare_alphas(
        "DT__uni+bi", (1, 2), contrast_alpha=1e-2,
        Xtr=Xtr_text, ytr=ytr, Xte=Xte_text, yte=yte
    )

    results = pd.concat([df_unibi, df_uni], axis=0).reset_index(drop=True)
    results_path = out / "dt_results.csv"
    results.to_csv(results_path, index=False)

    # Save per-alpha confusion matrices
    for tag, yhat in {**preds_uni, **preds_unibi}.items():
        cm = confusion_matrix(yte, yhat, labels=[NEG_LABEL, POS_LABEL])
        _save_confusion(cm, out / f"dt.confusion_matrix__{tag.replace('__', '_')}.csv")

    # Print disagreement counts
    total = int(test_mask.sum())

    # Choose better family by CV best score recorded in rows
    cv_scores = results.groupby(results["model"].str.contains("DT__uni+bi")).apply(
        lambda g: g["cv_best_score_macroF1"].max()
    )
    best_is_unibi = (cv_scores.get(True, -1) >= cv_scores.get(False, -1))
    best_ngr = (1, 2) if best_is_unibi else (1, 1)

    # Save top terms (directional log-odds)
    _save_log_odds_top_terms(Xtr_text, ytr, best_ngr, out)

    # Persist the best pipeline (by test F1 among all rows)
    best_row = results.sort_values("f1", ascending=False).iloc[0]
    use_pipe = pipe_unibi if "uni+bi" in best_row["model"] else pipe_uni
    dump(use_pipe, out / "dt_best_pipeline.joblib")

    return results_path
