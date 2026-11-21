"""SQLite-cached, retry-resilient DBpedia SPARQL client."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_ENDPOINT = "https://dbpedia.org/sparql"
DEFAULT_USER_AGENT = "kg-fact-checker/1.0 (FactBench-GERBIL)"


@dataclass
class KGSignals:
    direct_match: bool = False
    inverse_match: bool = False
    subject_out_degree: int = 0
    subject_in_degree: int = 0
    object_in_degree: int = 0
    object_out_degree: int = 0
    shared_neighbors_count: int = 0
    two_hop_paths_count: int = 0
    predicate_subject_count: int = 0
    predicate_object_count: int = 0
    subject_has_types: bool = False
    object_has_types: bool = False
    type_compatible: bool = False
    offline_or_empty: bool = True


class SPARQLCache:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS sparql_cache ("
                    "k TEXT PRIMARY KEY, payload TEXT NOT NULL, ts REAL NOT NULL)"
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def key(endpoint: str, query: str) -> str:
        return hashlib.sha256(f"{endpoint}\n{query}".encode("utf-8")).hexdigest()

    def get(self, endpoint: str, query: str) -> Optional[Dict[str, Any]]:
        k = self.key(endpoint, query)
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            try:
                row = conn.execute(
                    "SELECT payload FROM sparql_cache WHERE k = ?", (k,)
                ).fetchone()
                return json.loads(row[0]) if row else None
            finally:
                conn.close()

    def put(self, endpoint: str, query: str, data: Dict[str, Any]) -> None:
        k = self.key(endpoint, query)
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO sparql_cache (k, payload, ts) VALUES (?, ?, ?)",
                    (k, json.dumps(data), time.time()),
                )
                conn.commit()
            finally:
                conn.close()


class DBpediaClient:
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        cache_dir: str | Path = "cache",
        timeout: float = 12.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ) -> None:
        self.endpoint = endpoint
        self.cache = SPARQLCache(Path(cache_dir) / "sparql_cache.sqlite")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    @staticmethod
    def _is_uri(term: str) -> bool:
        t = term.strip()
        return t.startswith("http://") or t.startswith("https://")

    @staticmethod
    def _term(term: str) -> str:
        t = term.strip()
        if t.startswith("http://") or t.startswith("https://"):
            return f"<{t}>"
        escaped = t.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def query_json(self, sparql: str) -> Optional[Dict[str, Any]]:
        cached = self.cache.get(self.endpoint, sparql)
        if cached is not None:
            return cached

        url = f"{self.endpoint}?{urlencode({'query': sparql, 'format': 'application/sparql-results+json'})}"
        for attempt in range(self.max_retries + 1):
            req = Request(
                url,
                headers={
                    "User-Agent": DEFAULT_USER_AGENT,
                    "Accept": "application/sparql-results+json",
                },
            )
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                    if isinstance(data, dict):
                        self.cache.put(self.endpoint, sparql, data)
                        return data
            except HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2**attempt))
                    continue
                break
            except (URLError, TimeoutError, OSError, json.JSONDecodeError):
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2**attempt))
                    continue
                break
        return None

    @staticmethod
    def _int_val(data: Optional[Dict[str, Any]], var: str) -> int:
        if not data:
            return 0
        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            return 0
        try:
            return max(0, int(float(bindings[0].get(var, {}).get("value", 0))))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _ask_val(data: Optional[Dict[str, Any]]) -> bool:
        if not data:
            return False
        if "boolean" in data:
            return bool(data["boolean"])
        return len(data.get("results", {}).get("bindings", [])) > 0

    def _query_match_and_degrees(
        self, subject: str, predicate: str, obj: str
    ) -> tuple[bool, bool, int, int, int, int, int, int, bool]:
        s_uri, o_uri = self._is_uri(subject), self._is_uri(obj)
        s_t, p_t, o_t = self._term(subject), self._term(predicate), self._term(obj)
        responded = False

        r_dir = self.query_json(f"ASK WHERE {{ {s_t} {p_t} {o_t} . }}")
        responded = responded or (r_dir is not None)
        direct_match = self._ask_val(r_dir)

        inverse_match = False
        if s_uri and o_uri:
            r_inv = self.query_json(f"ASK WHERE {{ {o_t} {p_t} {s_t} . }}")
            responded = responded or (r_inv is not None)
            inverse_match = self._ask_val(r_inv)

        subj_out = subj_in = pred_subj = 0
        if s_uri:
            r_s = self.query_json(
                f"SELECT (COUNT(DISTINCT ?p1) AS ?outDeg) (COUNT(DISTINCT ?s2) AS ?inDeg) "
                f"(COUNT(DISTINCT ?s3) AS ?pSubj) WHERE {{ "
                f"{{ {s_t} ?p1 ?y1 . }} UNION {{ ?s2 ?p2 {s_t} . }} "
                f"OPTIONAL {{ ?s3 {p_t} ?o3 . }} }}"
            )
            if r_s is not None:
                responded = True
                subj_out = self._int_val(r_s, "outDeg")
                subj_in = self._int_val(r_s, "inDeg")
                pred_subj = self._int_val(r_s, "pSubj")

        obj_in = obj_out = pred_obj = 0
        if o_uri:
            r_o = self.query_json(
                f"SELECT (COUNT(DISTINCT ?s1) AS ?inDeg) (COUNT(DISTINCT ?p2) AS ?outDeg) "
                f"(COUNT(DISTINCT ?o3) AS ?pObj) WHERE {{ "
                f"{{ ?s1 ?p1 {o_t} . }} UNION {{ {o_t} ?p2 ?y2 . }} "
                f"OPTIONAL {{ ?s3 {p_t} ?o3 . }} }}"
            )
            if r_o is not None:
                responded = True
                obj_in = self._int_val(r_o, "inDeg")
                obj_out = self._int_val(r_o, "outDeg")
                pred_obj = self._int_val(r_o, "pObj")

        return (
            direct_match,
            inverse_match,
            subj_out,
            subj_in,
            obj_in,
            obj_out,
            pred_subj,
            pred_obj,
            responded,
        )
    
    def _query_neighborhood_and_types(
        self, subject: str, predicate: str, obj: str
    ) -> tuple[int, int, bool, bool, bool, bool]:
        s_uri, o_uri, p_uri = self._is_uri(subject), self._is_uri(obj), self._is_uri(predicate)
        s_t, p_t, o_t = self._term(subject), self._term(predicate), self._term(obj)
        responded = False

        shared = two_hops = 0
        if s_uri and o_uri and subject != obj:
            r_sh = self.query_json(
                f"SELECT (COUNT(DISTINCT ?mid) AS ?shared) WHERE {{ "
                f"{{ {s_t} ?p1 ?mid . ?mid ?p2 {o_t} . }} "
                f"UNION {{ {s_t} ?p3 ?mid . {o_t} ?p4 ?mid . }} "
                f"UNION {{ ?mid ?p5 {s_t} . ?mid ?p6 {o_t} . }} }}"
            )
            if r_sh is not None:
                responded = True
                shared = self._int_val(r_sh, "shared")

            r_2h = self.query_json(
                f"SELECT (COUNT(*) AS ?hops) WHERE {{ "
                f"{{ {s_t} ?p1 ?mid . ?mid {p_t} {o_t} . }} "
                f"UNION {{ {s_t} {p_t} ?mid . ?mid ?p2 {o_t} . }} }} LIMIT 100"
            )
            if r_2h is not None:
                responded = True
                two_hops = self._int_val(r_2h, "hops")

        s_types, o_types, expected_ranges = set(), set(), set()
        if s_uri:
            r_st = self.query_json(f"SELECT DISTINCT ?t WHERE {{ {s_t} a ?t . }} LIMIT 25")
            if r_st is not None:
                responded = True
                s_types = {b["t"]["value"] for b in r_st.get("results", {}).get("bindings", []) if "t" in b}
        if o_uri:
            r_ot = self.query_json(f"SELECT DISTINCT ?t WHERE {{ {o_t} a ?t . }} LIMIT 25")
            if r_ot is not None:
                responded = True
                o_types = {b["t"]["value"] for b in r_ot.get("results", {}).get("bindings", []) if "t" in b}
        if p_uri:
            r_rg = self.query_json(
                f"SELECT DISTINCT ?rg WHERE {{ {p_t} <http://www.w3.org/2000/01/rdf-schema#range> ?rg . }} LIMIT 20"
            )
            if r_rg is not None:
                responded = True
                expected_ranges = {b["rg"]["value"] for b in r_rg.get("results", {}).get("bindings", []) if "rg" in b}

        type_compat = (
            bool(expected_ranges & o_types) if (expected_ranges and o_types) else bool(s_types and o_types)
        )
        return shared, two_hops, bool(s_types), bool(o_types), type_compat, responded

    def get_triple_signals(self, subject: str, predicate: str, obj: str) -> KGSignals:
        """Assemble cached KG signals for a candidate (subject, predicate, object) triple."""
        key = f"signals::{subject}||{predicate}||{obj}"
        cached = self.cache.get(self.endpoint, key)
        if cached is not None:
            return KGSignals(**cached)

        direct_match, inverse_match, subj_out, subj_in, obj_in, obj_out, pred_subj, pred_obj, ok_a = (
            self._query_match_and_degrees(subject, predicate, obj)
        )
        shared, two_hops, has_s_t, has_o_t, type_compat, ok_b = self._query_neighborhood_and_types(
            subject, predicate, obj
        )
        signals = KGSignals(
            direct_match=direct_match,
            inverse_match=inverse_match,
            subject_out_degree=subj_out,
            subject_in_degree=subj_in,
            object_in_degree=obj_in,
            object_out_degree=obj_out,
            shared_neighbors_count=shared,
            two_hop_paths_count=two_hops,
            predicate_subject_count=pred_subj,
            predicate_object_count=pred_obj,
            subject_has_types=has_s_t,
            object_has_types=has_o_t,
            type_compatible=type_compat,
            offline_or_empty=not (ok_a or ok_b),
        )
        self.cache.put(self.endpoint, key, signals.__dict__)
        return signals
