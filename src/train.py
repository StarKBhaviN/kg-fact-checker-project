"""Train a fact-checking model with Stratified K-Fold CV and save artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (
    FEATURE_COLUMNS,
    FeatureEncoder,
    build_feature_matrix,
    fetch_signals_batch,
)
from src.io_rdf import parse_nt_statements
from src.kg_client import DBpediaClient


def _estimator(n_rows: int) -> Pipeline:
    if n_rows < 30:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ]
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=120,
                    learning_rate=0.06,
                    max_depth=4,
                    min_samples_leaf=5,
                    random_state=42,
                ),
            ),
        ]
    )
    def run_training(
    input_path: str | Path,
    model_out: str | Path = "models/fact_checker.joblib",
    cache_dir: str | Path = "cache",
    cv_folds: int = 5,
    offline: bool = False,
) -> dict:
    statements = parse_nt_statements(input_path)
    y = np.array([1 if (s.label is not None and s.label >= 0.5) else 0 for s in statements], dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise ValueError("Training set must include both positive and negative truth labels.")

    kg_client = None if offline else DBpediaClient(cache_dir=cache_dir)
    signals = fetch_signals_batch(statements, kg_client)

    folds = max(2, min(cv_folds, int(np.min(np.bincount(y))), len(statements)))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    oof = np.zeros(len(statements), dtype=np.float64)
    for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
        enc = FeatureEncoder(smoothing=10.0).fit([statements[i] for i in tr_idx], y[tr_idx].astype(float))
        x_tr = build_feature_matrix([statements[i] for i in tr_idx], [signals[i] for i in tr_idx], enc)
        x_va = build_feature_matrix([statements[i] for i in va_idx], [signals[i] for i in va_idx], enc)
        clf = _estimator(len(tr_idx)).fit(x_tr, y[tr_idx])
        oof[va_idx] = np.clip(clf.predict_proba(x_va)[:, 1], 0.0, 1.0)

    auc = float(roc_auc_score(y, oof))
    acc = float(accuracy_score(y, (oof >= 0.5).astype(int)))
    ll = float(log_loss(y, np.clip(oof, 1e-6, 1.0 - 1e-6)))
    print(f"Samples: {len(statements)} | Folds: {folds} | OOF ROC AUC: {auc:.4f} | Acc: {acc:.4f} | LogLoss: {ll:.4f}")
    full_enc = FeatureEncoder(smoothing=10.0).fit(statements, y.astype(float))
    x_full = build_feature_matrix(statements, signals, full_enc)
    final_clf = _estimator(len(statements)).fit(x_full, y)

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_clf,
            "encoder": full_enc,
            "feature_columns": FEATURE_COLUMNS,
            "cv_metrics": {"roc_auc": auc, "accuracy": acc, "log_loss": ll, "folds": folds},
        },
        out_path,
    )
    print(f"Saved model to {out_path}")
    return {"roc_auc": auc, "accuracy": acc, "log_loss": ll}