"""SQLite-cached, retry-resilient DBpedia SPARQL client."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Set
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
                conn.execute("PRAGMA journal_mode=WAL;")
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
        timeout: float = 6.0,
        max_retries: int = 1,
        backoff_base: float = 0.5,
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

    def query_json(self, sparql: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        cached = self.cache.get(self.endpoint, sparql)
        if cached is not None:
            return cached

        req_timeout = timeout if timeout is not None else self.timeout
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
                with urlopen(req, timeout=req_timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                    if isinstance(data, dict):
                        self.cache.put(self.endpoint, sparql, data)
                        return data
            except HTTPError as exc:
                if exc.code == 429 and attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2**attempt))
                    continue
                break
            except (URLError, TimeoutError, OSError, json.JSONDecodeError):
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

    def get_entity_degrees(self, entity: str) -> Tuple[int, int]:
        if not self._is_uri(entity):
            return 0, 0
        k = f"entity_deg::{entity}"
        cached = self.cache.get(self.endpoint, k)
        if cached is not None:
            return cached.get("out", 0), cached.get("in", 0)
        e_t = self._term(entity)
        r = self.query_json(
            f"SELECT (COUNT(DISTINCT ?p) AS ?outDeg) (COUNT(DISTINCT ?s) AS ?inDeg) WHERE {{ "
            f"{{ {e_t} ?p ?o . }} UNION {{ ?s ?p2 {e_t} . }} }}",
            timeout=5.0,
        )
        out_deg = self._int_val(r, "outDeg") if r else 0
        in_deg = self._int_val(r, "inDeg") if r else 0
        self.cache.put(self.endpoint, k, {"out": out_deg, "in": in_deg})
        return out_deg, in_deg

    def get_predicate_counts(self, predicate: str) -> Tuple[int, int]:
        if not self._is_uri(predicate):
            return 0, 0
        k = f"pred_counts::{predicate}"
        cached = self.cache.get(self.endpoint, k)
        if cached is not None:
            return cached.get("subj", 0), cached.get("obj", 0)
        p_t = self._term(predicate)
        r = self.query_json(
            f"SELECT (COUNT(DISTINCT ?s) AS ?pSubj) (COUNT(DISTINCT ?o) AS ?pObj) WHERE {{ ?s {p_t} ?o . }}",
            timeout=5.0,
        )
        p_s = self._int_val(r, "pSubj") if r else 0
        p_o = self._int_val(r, "pObj") if r else 0
        self.cache.put(self.endpoint, k, {"subj": p_s, "obj": p_o})
        return p_s, p_o

    def get_predicate_ranges(self, predicate: str) -> Set[str]:
        if not self._is_uri(predicate):
            return set()
        k = f"pred_range::{predicate}"
        cached = self.cache.get(self.endpoint, k)
        if cached is not None:
            return set(cached.get("ranges", []))
        p_t = self._term(predicate)
        r = self.query_json(
            f"SELECT DISTINCT ?rg WHERE {{ {p_t} <http://www.w3.org/2000/01/rdf-schema#range> ?rg . }} LIMIT 20",
            timeout=4.0,
        )
        ranges = (
            [b["rg"]["value"] for b in r.get("results", {}).get("bindings", []) if "rg" in b]
            if r
            else []
        )
        self.cache.put(self.endpoint, k, {"ranges": ranges})
        return set(ranges)

    def get_entity_types(self, entity: str) -> Set[str]:
        if not self._is_uri(entity):
            return set()
        k = f"entity_types::{entity}"
        cached = self.cache.get(self.endpoint, k)
        if cached is not None:
            return set(cached.get("types", []))
        e_t = self._term(entity)
        r = self.query_json(
            f"SELECT DISTINCT ?t WHERE {{ {e_t} a ?t . }} LIMIT 25",
            timeout=4.0,
        )
        types = (
            [b["t"]["value"] for b in r.get("results", {}).get("bindings", []) if "t" in b]
            if r
            else []
        )
        self.cache.put(self.endpoint, k, {"types": types})
        return set(types)

    def get_triple_signals(self, subject: str, predicate: str, obj: str) -> KGSignals:
        """Assemble cached KG signals for a candidate (subject, predicate, object) triple."""
        key = f"signals::{subject}||{predicate}||{obj}"
        cached = self.cache.get(self.endpoint, key)
        if cached is not None:
            return KGSignals(**cached)

        s_uri = self._is_uri(subject)
        o_uri = self._is_uri(obj)
        p_uri = self._is_uri(predicate)
        s_t = self._term(subject)
        p_t = self._term(predicate)
        o_t = self._term(obj)
        responded = False

        # 1. Direct match ASK
        r_dir = self.query_json(f"ASK WHERE {{ {s_t} {p_t} {o_t} . }}", timeout=5.0)
        responded = responded or (r_dir is not None)
        direct_match = self._ask_val(r_dir)

        # 2. Inverse match ASK
        inverse_match = False
        if s_uri and o_uri:
            r_inv = self.query_json(f"ASK WHERE {{ {o_t} {p_t} {s_t} . }}", timeout=5.0)
            responded = responded or (r_inv is not None)
            inverse_match = self._ask_val(r_inv)

        # 3. Entity degrees & Predicate counts (cached by entity / predicate)
        subj_out, subj_in = self.get_entity_degrees(subject) if s_uri else (0, 0)
        obj_in, obj_out = self.get_entity_degrees(obj) if o_uri else (0, 0)
        pred_subj, pred_obj = self.get_predicate_counts(predicate) if p_uri else (0, 0)

        # 4. Shared neighbors & 2-hop paths (only if not identical and both URIs)
        shared = two_hops = 0
        if s_uri and o_uri and subject != obj:
            r_2h = self.query_json(
                f"SELECT (COUNT(*) AS ?hops) WHERE {{ "
                f"{{ {s_t} ?p1 ?mid . ?mid {p_t} {o_t} . }} "
                f"UNION {{ {s_t} {p_t} ?mid . ?mid ?p2 {o_t} . }} }} LIMIT 50",
                timeout=5.0,
            )
            if r_2h is not None:
                responded = True
                two_hops = self._int_val(r_2h, "hops")

            r_sh = self.query_json(
                f"SELECT (COUNT(DISTINCT ?mid) AS ?shared) WHERE {{ "
                f"{{ {s_t} ?p1 ?mid . ?mid ?p2 {o_t} . }} "
                f"UNION {{ {s_t} ?p3 ?mid . {o_t} ?p4 ?mid . }} }} LIMIT 50",
                timeout=5.0,
            )
            if r_sh is not None:
                responded = True
                shared = self._int_val(r_sh, "shared")

        # 5. Types & Type compatibility (cached by entity / predicate)
        s_types = self.get_entity_types(subject) if s_uri else set()
        o_types = self.get_entity_types(obj) if o_uri else set()
        expected_ranges = self.get_predicate_ranges(predicate) if p_uri else set()
        type_compat = (
            bool(expected_ranges & o_types)
            if (expected_ranges and o_types)
            else bool(s_types and o_types)
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
            subject_has_types=bool(s_types),
            object_has_types=bool(o_types),
            type_compatible=type_compat,
            offline_or_empty=not (responded or direct_match or inverse_match or subj_out > 0 or obj_in > 0),
        )
        self.cache.put(self.endpoint, key, signals.__dict__)
        return signals
