"""RDF I/O for reified FactBench .nt inputs and GERBIL Turtle (.ttl) outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

SWC = Namespace("http://swc2017.aksw.org/")
HAS_TRUTH_VALUE = SWC.hasTruthValue


@dataclass(frozen=True)
class FactStatement:
    stmt_uri: str
    subject: str
    predicate: str
    object: str
    label: Optional[float] = None


def _parse_truth_literal(literal: Literal) -> Optional[float]:
    try:
        val = float(literal.toPython())
    except (ValueError, TypeError):
        return None
    if not math.isfinite(val):
        return None
    return max(0.0, min(1.0, val))


def parse_nt_statements(nt_path: str | Path) -> List[FactStatement]:
    """Parse reified rdf:Statement triples (with optional hasTruthValue) from .nt."""
    path = Path(nt_path)
    graph = Graph()
    graph.parse(str(path), format="nt")

    stmt_uris: List[URIRef] = []
    seen: Set[URIRef] = set()
    for s in graph.subjects(RDF.type, RDF.Statement):
        if isinstance(s, URIRef) and s not in seen:
            seen.add(s)
            stmt_uris.append(s)

    truth_map: dict[URIRef, float] = {}
    for s, _, o in graph.triples((None, HAS_TRUTH_VALUE, None)):
        if isinstance(s, URIRef) and isinstance(o, Literal):
            parsed = _parse_truth_literal(o)
            if parsed is not None:
                truth_map[s] = parsed

    records: List[FactStatement] = []
    for stmt in stmt_uris:
        subj = graph.value(stmt, RDF.subject)
        pred = graph.value(stmt, RDF.predicate)
        obj = graph.value(stmt, RDF.object)
        if subj is None or pred is None or obj is None:
            raise ValueError(f"Incomplete statement {stmt}")
        records.append(
            FactStatement(
                stmt_uri=str(stmt),
                subject=str(subj),
                predicate=str(pred),
                object=str(obj),
                label=truth_map.get(stmt),
            )
        )
    return records


def format_truth_score(score: float) -> str:
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        val = 0.5
    else:
        val = max(0.0, min(1.0, float(score)))
    return f'"{val:.4f}"^^<{XSD.double}>'


def write_result_ttl(
    stmt_uris: Sequence[str],
    scores: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Write one hasTruthValue triple per statement URI in valid Turtle."""
    if len(stmt_uris) != len(scores):
        raise ValueError("stmt_uris and scores must have equal length")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .", ""]
    for uri_str, raw_score in zip(stmt_uris, scores):
        uri = URIRef(str(uri_str).strip())
        lines.append(f"{uri.n3()} <{HAS_TRUTH_VALUE}> {format_truth_score(float(raw_score))} .")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
