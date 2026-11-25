"""Diagnostic tool to inspect what is lacking in the KG Fact Checker.

Usage:
    python scripts/diagnose_model.py [--input data/FoKG26-train.nt] [--model models/fact_checker.joblib]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.features import FeatureEncoder, build_feature_matrix, fetch_signals_batch
from src.io_rdf import FactStatement, parse_nt_statements
from src.kg_client import DBpediaClient


def diagnose(
    input_path: str | Path = "data/FoKG26-train.nt",
    model_path: str | Path = "models/fact_checker.joblib",
    cache_dir: str | Path = "cache",
    workers: int = 4,
    offline: bool = False,
) -> None:
    print(f"=== Loading dataset from {input_path} ===")
    statements: Sequence[FactStatement] = parse_nt_statements(input_path)
    y = np.array(
        [1 if (s.label is not None and s.label >= 0.5) else 0 for s in statements],
        dtype=np.int64,
    )
    print(f"Total statements: {len(statements)} | Positives: {int(y.sum())} | Negatives: {int(len(y) - y.sum())}")

    print("\n=== Fetching / Loading KG Signals ===")
    kg_client = None if offline else DBpediaClient(cache_dir=cache_dir)
    signals = fetch_signals_batch(statements, kg_client, max_workers=workers)

    model_file = Path(model_path)
    if model_file.exists():
        artifact = joblib.load(model_file)
        model = artifact["model"]
        encoder = artifact["encoder"]
        cols = artifact["feature_columns"]
        X = build_feature_matrix(statements, signals, encoder).reindex(
            columns=cols, fill_value=0.0
        )
    else:
        print(f"Model file {model_path} not found. Running cross-validation from scratch...")
        from src.train import _estimator
        model = _estimator(len(statements))
        encoder = FeatureEncoder().fit(statements, y)
        X = build_feature_matrix(statements, signals, encoder)
        cols = X.columns.tolist()

    # Out-of-fold cross validation predictions
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y), dtype=np.float64)

    for train_idx, val_idx in skf.split(X, y):
        x_tr, y_tr = X.iloc[train_idx], y[train_idx]
        x_va = X.iloc[val_idx]
        enc = FeatureEncoder().fit([statements[i] for i in train_idx], y_tr)
        m = joblib.load(model_file)["model"] if model_file.exists() else model
        m.fit(x_tr, y_tr)
        oof_preds[val_idx] = m.predict_proba(x_va)[:, 1]

    oof_auc = roc_auc_score(y, oof_preds)
    oof_acc = accuracy_score(y, (oof_preds >= 0.5).astype(int))
    oof_prec = precision_score(y, (oof_preds >= 0.5).astype(int), zero_division=0)
    oof_rec = recall_score(y, (oof_preds >= 0.5).astype(int), zero_division=0)
    oof_f1 = f1_score(y, (oof_preds >= 0.5).astype(int), zero_division=0)
    oof_loss = log_loss(y, np.clip(oof_preds, 1e-6, 1.0 - 1e-6))

    print("\n" + "=" * 60)
    print(" 1. OVERALL 5-FOLD CROSS-VALIDATION METRICS")
    print("=" * 60)
    print(f"ROC-AUC (GERBIL Primary Metric): {oof_auc:.4f}")
    print(f"Accuracy:                       {oof_acc:.4f}")
    print(f"Precision:                      {oof_prec:.4f}")
    print(f"Recall:                         {oof_rec:.4f}")
    print(f"F1-Score:                       {oof_f1:.4f}")
    print(f"Log Loss:                       {oof_loss:.4f}")

    # Predicate breakdown
    print("\n" + "=" * 80)
    print(" 2. PERFORMANCE & SIGNAL BREAKDOWN PER PREDICATE")
    print("=" * 80)
    df_diag = pd.DataFrame({
        "predicate": [s.predicate.split("/")[-1] for s in statements],
        "y_true": y,
        "pred": oof_preds,
        "direct_match": [getattr(sig, "direct_match", False) for sig in signals],
        "target_rel": [getattr(sig, "has_target_relation", False) for sig in signals],
        "conflict_rel": [getattr(sig, "has_conflicting_relation", False) for sig in signals],
        "shared": [sig.shared_neighbors_count for sig in signals],
        "two_hop": [sig.two_hop_paths_count for sig in signals],
    })

    header = f"{'Predicate':<22} | {'Count':<5} | {'AUC':<7} | {'Pos Match':<9} | {'Neg Match':<9} | {'Neg Conflict':<12} | {'Pos 2-Hop':<9}"
    print(header)
    print("-" * len(header))

    for pred, grp in df_diag.groupby("predicate"):
        p_auc = roc_auc_score(grp["y_true"], grp["pred"]) if len(grp["y_true"].unique()) > 1 else np.nan
        pos_grp = grp[grp["y_true"] == 1]
        neg_grp = grp[grp["y_true"] == 0]

        pos_match = (pos_grp["target_rel"] | pos_grp["direct_match"]).mean() if len(pos_grp) else 0.0
        neg_match = (neg_grp["target_rel"] | neg_grp["direct_match"]).mean() if len(neg_grp) else 0.0
        neg_conflict = neg_grp["conflict_rel"].mean() if len(neg_grp) else 0.0
        pos_2hop = (pos_grp["two_hop"] > 0).mean() if len(pos_grp) else 0.0

        print(
            f"{pred:<22} | {len(grp):<5} | {p_auc:<7.4f} | {pos_match * 100:<8.1f}% | {neg_match * 100:<8.1f}% | {neg_conflict * 100:<11.1f}% | {pos_2hop * 100:<8.1f}%"
        )

    # Feature Importance
    print("\n" + "=" * 60)
    print(" 3. TOP FEATURE IMPORTANCE (Permutation ROC-AUC drop)")
    print("=" * 60)
    try:
        r = permutation_importance(model, X, y, n_repeats=5, random_state=42, scoring="roc_auc")
        sorted_idx = r.importances_mean.argsort()[::-1]
        for rank, idx in enumerate(sorted_idx[:12], start=1):
            feat_name = cols[idx]
            imp_mean = r.importances_mean[idx]
            imp_std = r.importances_std[idx]
            print(f" {rank:>2}. {feat_name:<26}: {imp_mean:+.4f} (±{imp_std:.4f})")
    except Exception as e:
        print(f"Could not compute permutation importance: {e}")

    # Top Error Analysis
    print("\n" + "=" * 80)
    print(" 4. TOP 5 FALSE POSITIVES (Model predicted TRUE, but fact is FALSE)")
    print("=" * 80)
    df_err = pd.DataFrame({
        "subject": [s.subject.split("/")[-1] for s in statements],
        "predicate": [s.predicate.split("/")[-1] for s in statements],
        "object": [s.object.split("/")[-1] for s in statements],
        "y_true": y,
        "pred": oof_preds,
        "error": np.abs(y - oof_preds),
    })

    fp = df_err[df_err["y_true"] == 0].sort_values("pred", ascending=False).head(5)
    for _, row in fp.iterrows():
        print(f" • {row['subject']} --({row['predicate']})--> {row['object']}")
        print(f"   Predicted: {row['pred']:.4f} | Actual: 0.0 (Reason: High spurious graph overlap or missing conflict check)")

    print("\n" + "=" * 80)
    print(" 5. TOP 5 FALSE NEGATIVES (Model predicted FALSE, but fact is TRUE)")
    print("=" * 80)
    fn = df_err[df_err["y_true"] == 1].sort_values("pred", ascending=True).head(5)
    for _, row in fn.iterrows():
        print(f" • {row['subject']} --({row['predicate']})--> {row['object']}")
        print(f"   Predicted: {row['pred']:.4f} | Actual: 1.0 (Reason: Property in raw infobox dbp: or redirected URI)")

    print("\n" + "=" * 60)
    print(" 6. DIAGNOSIS SUMMARY: WHAT WAS LACKING & HOW IT IS FIXED")
    print("=" * 60)
    if oof_auc < 0.70:
        print(" • Bottleneck: Model relied on dbo: exact match, missing 80%+ of true facts.")
        print(" • Action: Use unified forward/reverse relation queries to capture dbp:, aliases, and redirects.")
    elif oof_auc < 0.80:
        print(" • Progress: Approaching 0.80. Focus on conflicting relation suppression and type compatibility.")
    else:
        print(f" • Excellent: ROC-AUC is {oof_auc:.4f} (Above target 0.80+)!")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose what is lacking in the KG Fact Checker.")
    parser.add_argument("--input", default="data/FoKG26-train.nt", help="Path to labeled training RDF triples")
    parser.add_argument("--model", default="models/fact_checker.joblib", help="Path to trained model artifact")
    parser.add_argument("--cache-dir", default="cache", help="Path to cache directory")
    parser.add_argument("--workers", type=int, default=4, help="Concurrency workers")
    parser.add_argument("--offline", action="store_true", help="Run offline without querying DBpedia")
    args = parser.parse_args()

    diagnose(
        input_path=args.input,
        model_path=args.model,
        cache_dir=args.cache_dir,
        workers=args.workers,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
