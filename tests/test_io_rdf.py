"""Unit tests for .nt parsing, Turtle writing and validate_turtle_output."""

from pathlib import Path

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from src.io_rdf import (
    HAS_TRUTH_VALUE,
    parse_nt_statements,
    validate_turtle_output,
    write_result_ttl,
)

SAMPLE_NT = """<http://example.org/stmt_1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> .
<http://example.org/stmt_1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject> <http://dbpedia.org/resource/Berlin> .
<http://example.org/stmt_1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate> <http://dbpedia.org/ontology/country> .
<http://example.org/stmt_1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#object> <http://dbpedia.org/resource/Germany> .
<http://example.org/stmt_1> <http://swc2017.aksw.org/hasTruthValue> "1.0"^^<http://www.w3.org/2001/XMLSchema#double> .
<http://example.org/stmt_2> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> .
<http://example.org/stmt_2> <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject> <http://dbpedia.org/resource/Berlin> .
<http://example.org/stmt_2> <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate> <http://dbpedia.org/ontology/country> .
<http://example.org/stmt_2> <http://www.w3.org/1999/02/22-rdf-syntax-ns#object> <http://dbpedia.org/resource/France> .
"""


def test_parse_and_validate_roundtrip(tmp_path: Path) -> None:
    nt_file = tmp_path / "data.nt"
    nt_file.write_text(SAMPLE_NT, encoding="utf-8")
    stmts = parse_nt_statements(nt_file)
    assert len(stmts) == 2
    assert stmts[0].label == pytest.approx(1.0)
    assert stmts[1].label is None

    ttl_file = tmp_path / "result.ttl"
    write_result_ttl([s.stmt_uri for s in stmts], [1.5, -0.2], ttl_file)
    summary = validate_turtle_output(ttl_file, expected_stmt_uris=[s.stmt_uri for s in stmts])
    assert summary["triple_count"] == 2
    assert summary["min_score"] == pytest.approx(0.0)
    assert summary["max_score"] == pytest.approx(1.0)


def test_validate_catches_missing_and_out_of_range(tmp_path: Path) -> None:
    s1 = URIRef("http://example.org/stmt_1")
    s2 = URIRef("http://example.org/stmt_2")
    partial = tmp_path / "partial.ttl"
    write_result_ttl([str(s1)], [0.7], partial)
    with pytest.raises(ValueError, match="Missing 1 expected statement URIs"):
        validate_turtle_output(partial, expected_stmt_uris=[str(s1), str(s2)])

    bad = tmp_path / "bad.ttl"
    g = Graph()
    g.add((s1, RDF.type, RDF.Statement))
    g.add((s1, HAS_TRUTH_VALUE, Literal("1.3", datatype=XSD.double)))
    g.serialize(destination=str(bad), format="turtle")
    with pytest.raises(ValueError, match="out of \\[0.0, 1.0\\]"):
        validate_turtle_output(bad, expected_stmt_uris=[str(s1)])