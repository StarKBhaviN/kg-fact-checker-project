"""Leak-free feature extraction combining KG signals and fold-local statistics."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
from urllib.parse import unquote

import numpy as np
import pandas as pd

from src.io_rdf import FactStatement
from src.kg_client import DBpediaClient, KGSignals

FEATURE_COLUMNS: List[str] = [
    "has_target_relation",
    "has_conflicting_relation",
    "has_other_relation",
    "direct_match",
    "property_match",
    "redirect_match",
    "any_direct_match",
    "inverse_match",
    "fwd_rel_count",
    "rev_rel_count",
    "verified_true_signal",
    "log1p_subj_out_deg",
    "log1p_subj_in_deg",
    "log1p_obj_in_deg",
    "log1p_obj_out_deg",
    "log1p_shared_neighbors",
    "log1p_two_hop_paths",
    "has_2hop",
    "has_shared",
    "direct_or_2hop",
    "token_jaccard",
    "graph_jaccard",
    "connectivity_composite",
    "log1p_pred_subj_cnt",
    "log1p_pred_obj_cnt",
    "subj_has_types",
    "obj_has_types",
    "type_compatible",
    "subj_is_uri",
    "obj_is_uri",
    "subj_obj_identical",
    "pred_freq_train",
    "pred_target_mean_oof",
    "subj_freq_train",
    "obj_freq_train",
    "kg_data_available",
]


def _is_uri(v: str) -> bool:
    s = v.strip()
    return s.startswith("http://") or s.startswith("https://")


def _log1p(x: float) -> float:
    return float(math.log1p(x)) if math.isfinite(x) and x >= 0.0 else 0.0


@dataclass
class FeatureEncoder:
    smoothing: float = 10.0
    global_prior: float = 0.5
    total_rows: int = 0
    pred_counts: Dict[str, int] = field(default_factory=dict)
    pred_sums: Dict[str, float] = field(default_factory=dict)
    subj_counts: Dict[str, int] = field(default_factory=dict)
    obj_counts: Dict[str, int] = field(default_factory=dict)

    def fit(self, statements: Sequence[FactStatement], labels: Sequence[float]) -> "FeatureEncoder":
        self.total_rows = len(statements)
        if self.total_rows == 0:
            return self
        y = np.asarray(labels, dtype=np.float64)
        self.global_prior = float(np.mean(y[np.isfinite(y)])) if np.any(np.isfinite(y)) else 0.5
        self.pred_counts, self.pred_sums, self.subj_counts, self.obj_counts = {}, {}, {}, {}
        for stmt, val in zip(statements, y):
            yv = float(val) if math.isfinite(float(val)) else self.global_prior
            self.pred_counts[stmt.predicate] = self.pred_counts.get(stmt.predicate, 0) + 1
            self.pred_sums[stmt.predicate] = self.pred_sums.get(stmt.predicate, 0.0) + yv
            self.subj_counts[stmt.subject] = self.subj_counts.get(stmt.subject, 0) + 1
            self.obj_counts[stmt.object] = self.obj_counts.get(stmt.object, 0) + 1
        return self

    def transform_one(self, stmt: FactStatement, sig: KGSignals) -> Dict[str, float]:
        denom = max(1, self.total_rows)
        p_cnt = self.pred_counts.get(stmt.predicate, 0)
        p_sum = self.pred_sums.get(stmt.predicate, 0.0)

        target_rel = 1.0 if getattr(sig, "has_target_relation", False) else 0.0
        conflict_rel = 1.0 if getattr(sig, "has_conflicting_relation", False) else 0.0
        other_rel = 1.0 if getattr(sig, "has_other_relation", False) else 0.0
        fwd_cnt = float(getattr(sig, "fwd_rel_count", 0))
        rev_cnt = float(getattr(sig, "rev_rel_count", 0))

        dm = 1.0 if getattr(sig, "direct_match", False) else 0.0
        pm = 1.0 if getattr(sig, "property_match", False) else 0.0
        rm = 1.0 if getattr(sig, "redirect_match", False) else 0.0
        any_dm = 1.0 if (target_rel > 0 or dm > 0 or pm > 0 or rm > 0) else 0.0
        h2 = 1.0 if sig.two_hop_paths_count > 0 else 0.0
        hs = 1.0 if sig.shared_neighbors_count > 0 else 0.0

        # Text token overlap between subject and object URIs
        s_tokens = set(unquote(stmt.subject.split("/")[-1]).replace("_", " ").lower().split())
        o_tokens = set(unquote(stmt.object.split("/")[-1]).replace("_", " ").lower().split())
        token_jacc = len(s_tokens & o_tokens) / max(1, len(s_tokens | o_tokens))

        # Normalized graph Jaccard similarity
        subj_deg = _log1p(sig.subject_out_degree)
        obj_deg = _log1p(sig.object_in_degree)
        deg_sum = subj_deg + obj_deg
        graph_jacc = _log1p(sig.shared_neighbors_count) / (deg_sum + 1.0)
        conn_composite = 3.0 * any_dm + 1.5 * h2 + 0.5 * hs

        # Verified composite signal: rewards matching target relation, strongly penalizes FactBench conflict corruptions
        verified_sig = 2.0 * target_rel + 1.0 * any_dm - 2.5 * conflict_rel

        return {
            "has_target_relation": target_rel,
            "has_conflicting_relation": conflict_rel,
            "has_other_relation": other_rel,
            "direct_match": dm,
            "property_match": pm,
            "redirect_match": rm,
            "any_direct_match": any_dm,
            "inverse_match": 1.0 if sig.inverse_match else 0.0,
            "fwd_rel_count": fwd_cnt,
            "rev_rel_count": rev_cnt,
            "verified_true_signal": verified_sig,
            "log1p_subj_out_deg": subj_deg,
            "log1p_subj_in_deg": _log1p(sig.subject_in_degree),
            "log1p_obj_in_deg": obj_deg,
            "log1p_obj_out_deg": _log1p(sig.object_out_degree),
            "log1p_shared_neighbors": _log1p(sig.shared_neighbors_count),
            "log1p_two_hop_paths": _log1p(sig.two_hop_paths_count),
            "has_2hop": h2,
            "has_shared": hs,
            "direct_or_2hop": 1.0 if (dm > 0 or h2 > 0) else 0.0,
            "token_jaccard": token_jacc,
            "graph_jaccard": graph_jacc,
            "connectivity_composite": conn_composite,
            "log1p_pred_subj_cnt": _log1p(sig.predicate_subject_count),
            "log1p_pred_obj_cnt": _log1p(sig.predicate_object_count),
            "subj_has_types": 1.0 if sig.subject_has_types else 0.0,
            "obj_has_types": 1.0 if sig.object_has_types else 0.0,
            "type_compatible": 1.0 if sig.type_compatible else 0.0,
            "subj_is_uri": 1.0 if _is_uri(stmt.subject) else 0.0,
            "obj_is_uri": 1.0 if _is_uri(stmt.object) else 0.0,
            "subj_obj_identical": 1.0 if stmt.subject.strip() == stmt.object.strip() else 0.0,
            "pred_freq_train": p_cnt / denom,
            "pred_target_mean_oof": (p_sum + self.smoothing * self.global_prior) / (p_cnt + self.smoothing),
            "subj_freq_train": self.subj_counts.get(stmt.subject, 0) / denom,
            "obj_freq_train": self.obj_counts.get(stmt.object, 0) / denom,
            "kg_data_available": 0.0 if sig.offline_or_empty else 1.0,
        }


def fetch_signals_batch(
    statements: Sequence[FactStatement],
    kg_client: Optional[DBpediaClient],
    max_workers: int = 4,
    refresh: bool = False,
) -> List[KGSignals]:
    if kg_client is None or len(statements) == 0:
        return [KGSignals(offline_or_empty=True) for _ in statements]

    total = len(statements)
    print(f"Fetching KG signals for {total} statements (concurrency={max_workers})...", flush=True)

    # Pre-fetch predicate metadata first (ranges and counts) for the distinct predicates
    unique_preds = {s.predicate for s in statements if _is_uri(s.predicate)}
    for p in unique_preds:
        try:
            kg_client.get_predicate_ranges(p)
            kg_client.get_predicate_counts(p)
        except Exception:
            pass

    results: List[Optional[KGSignals]] = [None] * total

    def _worker(idx: int, stmt: FactStatement) -> tuple[int, KGSignals]:
        time.sleep(0.01)
        try:
            sig = kg_client.get_triple_signals(
                stmt.subject, stmt.predicate, stmt.object, refresh=refresh
            )
        except Exception:
            sig = KGSignals(offline_or_empty=True)
        return idx, sig

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, i, stmt) for i, stmt in enumerate(statements)]
        for fut in as_completed(futures):
            idx, sig = fut.result()
            results[idx] = sig
            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f"  [KG signals] {completed}/{total} ({completed / total * 100:.1f}%) processed", flush=True)

    return [sig if sig is not None else KGSignals(offline_or_empty=True) for sig in results]


def build_feature_matrix(
    statements: Sequence[FactStatement],
    signals_list: Sequence[KGSignals],
    encoder: FeatureEncoder,
) -> pd.DataFrame:
    if not statements:
        return pd.DataFrame(columns=FEATURE_COLUMNS, dtype=np.float64)
    rows = [encoder.transform_one(s, sig) for s, sig in zip(statements, signals_list)]
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)
