"""Unit tests for feature extraction and DBpedia client caching."""

from pathlib import Path
from unittest.mock import MagicMock

from src.features import FeatureEncoder, build_feature_matrix, fetch_signals_batch
from src.io_rdf import FactStatement
from src.kg_client import DBpediaClient, KGSignals, SPARQLCache


def test_sparql_cache_put_get(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    cache = SPARQLCache(db)
    assert cache.get("http://example.org", "query1") is None
    cache.put("http://example.org", "query1", {"head": {}, "boolean": True})
    res = cache.get("http://example.org", "query1")
    assert res == {"head": {}, "boolean": True}


def test_fetch_signals_offline() -> None:
    stmts = [
        FactStatement("uri1", "s1", "p1", "o1", 1.0),
        FactStatement("uri2", "s2", "p2", "o2", 0.0),
    ]
    signals = fetch_signals_batch(stmts, kg_client=None)
    assert len(signals) == 2
    assert signals[0].offline_or_empty is True
    assert signals[1].offline_or_empty is True


def test_encoder_and_matrix_construction() -> None:
    stmts = [
        FactStatement("uri1", "http://dbpedia.org/resource/A", "http://dbpedia.org/ontology/p", "http://dbpedia.org/resource/B", 1.0),
        FactStatement("uri2", "http://dbpedia.org/resource/C", "http://dbpedia.org/ontology/p", "http://dbpedia.org/resource/D", 0.0),
    ]
    sigs = [
        KGSignals(direct_match=True, subject_out_degree=10, offline_or_empty=False),
        KGSignals(direct_match=False, subject_out_degree=2, offline_or_empty=False),
    ]
    enc = FeatureEncoder().fit(stmts, [1.0, 0.0])
    df = build_feature_matrix(stmts, sigs, enc)
    assert len(df) == 2
    assert "direct_match" in df.columns
    assert df["direct_match"].tolist() == [1.0, 0.0]
    assert df["log1p_subj_out_deg"].iloc[0] > 0.0
