"""Generate truth-value scores in Turtle format and validate against expected URIs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import joblib

from src.features import FEATURE_COLUMNS, build_feature_matrix, fetch_signals_batch
from src.io_rdf import parse_nt_statements, validate_turtle_output, write_result_ttl
from src.kg_client import DBpediaClient


def run_prediction(
    input_path: str | Path,
    model_in: str | Path = "models/fact_checker.joblib",
    output_path: str | Path = "outputs/result.ttl",
    cache_dir: str | Path = "cache",
    offline: bool = False,
) -> Path:
    statements = parse_nt_statements(input_path)
    artifact = joblib.load(model_in)
    model = artifact["model"]
    encoder = artifact["encoder"]
    cols = artifact.get("feature_columns", FEATURE_COLUMNS)

    kg_client = None if offline else DBpediaClient(cache_dir=cache_dir)
    signals = fetch_signals_batch(statements, kg_client)
    x_test = build_feature_matrix(statements, signals, encoder).reindex(
        columns=cols, fill_value=0.0
    )

    probs = model.predict_proba(x_test)[:, 1]
    scores: List[float] = []
    for p in probs:
        v = float(p)
        if not math.isfinite(v):
            v = float(encoder.global_prior)
        scores.append(max(0.0, min(1.0, v)))

    uris = [s.stmt_uri for s in statements]
    out_file = write_result_ttl(uris, scores, output_path)
    summary = validate_turtle_output(out_file, expected_stmt_uris=uris)
    print(
        f"Wrote {summary['triple_count']} triples to {summary['path']} (range [{summary['min_score']:.4f}, {summary['max_score']:.4f}])"
    )
    return out_file


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Predict fact truth values into Turtle."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-in", default="models/fact_checker.joblib")
    parser.add_argument("--output", default="outputs/result.ttl")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    run_prediction(args.input, args.model_in, args.output, args.cache_dir, args.offline)


if __name__ == "__main__":
    main()
