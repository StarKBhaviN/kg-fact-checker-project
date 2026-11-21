"""Leak-free feature extraction combining KG signals and fold-local statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.io_rdf import FactStatement
from src.kg_client import DBpediaClient, KGSignals

FEATURE_COLUMNS: List[str] = [
    "direct_match",
    "inverse_match",
    "log1p_subj_out_deg",
    "log1p_subj_in_deg",
    "log1p_obj_in_deg",
    "log1p_obj_out_deg",
    "log1p_shared_neighbors",
    "log1p_two_hop_paths",
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
        return {
            "direct_match": 1.0 if sig.direct_match else 0.0,
            "inverse_match": 1.0 if sig.inverse_match else 0.0,
            "log1p_subj_out_deg": _log1p(sig.subject_out_degree),
            "log1p_subj_in_deg": _log1p(sig.subject_in_degree),
            "log1p_obj_in_deg": _log1p(sig.object_in_degree),
            "log1p_obj_out_deg": _log1p(sig.object_out_degree),
            "log1p_shared_neighbors": _log1p(sig.shared_neighbors_count),
            "log1p_two_hop_paths": _log1p(sig.two_hop_paths_count),
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
) -> List[KGSignals]:
    if kg_client is None:
        return [KGSignals(offline_or_empty=True) for _ in statements]
    out: List[KGSignals] = []
    for stmt in statements:
        try:
            out.append(kg_client.get_triple_signals(stmt.subject, stmt.predicate, stmt.object))
        except Exception:
            out.append(KGSignals(offline_or_empty=True))
    return out


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
