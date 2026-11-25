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
    property_match: bool = False
    redirect_match: bool = False
    inverse_match: bool = False
    has_target_relation: bool = False
    has_conflicting_relation: bool = False
    has_other_relation: bool = False
    fwd_rel_count: int = 0
    rev_rel_count: int = 0
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


ALIAS_MAP: Dict[str, Set[str]] = {
    "director": {"director", "directing"},
    "producer": {"producer", "producers", "executiveproducer"},
    "productioncompany": {
        "productioncompany",
        "productioncompanies",
        "studio",
        "production",
    },
    "starring": {"starring", "cast", "actor", "actors", "star"},
    "birthplace": {"birthplace", "birth", "born"},
    "deathplace": {"deathplace", "death", "died", "restingplace"},
    "city": {"city", "location", "headquarter", "headquarters", "campus"},
    "affiliation": {"affiliation", "affiliations", "association", "member"},
    "academicdiscipline": {
        "academicdiscipline",
        "discipline",
        "field",
        "fields",
    },
    "award": {
        "award",
        "awards",
        "honour",
        "honours",
        "prize",
        "recipient",
        "winner",
    },
    "chancellor": {"chancellor", "chancellors", "president", "rector"},
}

CONFLICT_MAP: Dict[str, Set[str]] = {
    "deathplace": {"birthplace", "birth"},
    "birthplace": {"deathplace", "death"},
    "director": {
        "writer",
        "starring",
        "producer",
        "editor",
        "cinematography",
        "musiccomposer",
    },
    "producer": {"director", "writer", "starring", "editor"},
    "starring": {
        "director",
        "producer",
        "writer",
        "composer",
        "musiccomposer",
    },
    "productioncompany": {"distributor", "distributors", "starring", "director", "writer"},
}


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
                if exc.code in (429, 503) and attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2**attempt))
                    continue
                break
            except (URLError, TimeoutError, OSError, json.JSONDecodeError):
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base * (1.5**attempt))
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

    def get_canonical_uri(self, uri: str) -> str:
        if not self._is_uri(uri):
            return uri
        k = f"canonical::{uri}"
        cached = self.cache.get(self.endpoint, k)
        if cached is not None:
            return cached.get("canonical", uri)
        u_t = self._term(uri)
        r = self.query_json(
            f"SELECT ?target WHERE {{ {u_t} <http://dbpedia.org/ontology/wikiPageRedirects> ?target . }} LIMIT 1",
            timeout=3.0,
        )
        canonical = uri
        if r:
            bindings = r.get("results", {}).get("bindings", [])
            if bindings and "target" in bindings[0]:
                canonical = bindings[0]["target"]["value"]
        self.cache.put(self.endpoint, k, {"canonical": canonical})
        return canonical

    def get_triple_signals(
        self, subject: str, predicate: str, obj: str, refresh: bool = False
    ) -> KGSignals:
        """Assemble cached KG signals for a candidate (subject, predicate, object) triple."""
        key = f"signals_v3::{subject}||{predicate}||{obj}"
        cached = self.cache.get(self.endpoint, key)
        if cached is not None and not refresh:
            valid_kwargs = {k: v for k, v in cached.items() if k in KGSignals.__dataclass_fields__}
            return KGSignals(**valid_kwargs)

        s_uri = self._is_uri(subject)
        o_uri = self._is_uri(obj)
        p_uri = self._is_uri(predicate)
        pred_name = predicate.split("/")[-1].lower() if p_uri else ""
        responded = False

        s_can = self.get_canonical_uri(subject) if s_uri else subject
        o_can = self.get_canonical_uri(obj) if o_uri else obj

        fwd_names: Set[str] = set()
        rev_names: Set[str] = set()

        if s_uri and o_uri:
            s_t = self._term(s_can)
            o_t = self._term(o_can)
            q_rel = f"""
            SELECT DISTINCT ?p_fwd ?p_rev WHERE {{
              {{ {s_t} ?p_fwd {o_t} . FILTER(?p_fwd != <http://dbpedia.org/ontology/wikiPageWikiLink>) }}
              UNION
              {{ {o_t} ?p_rev {s_t} . FILTER(?p_rev != <http://dbpedia.org/ontology/wikiPageWikiLink>) }}
            }} LIMIT 30
            """
            r_rel = self.query_json(q_rel, timeout=5.0)
            if r_rel is not None:
                responded = True
                bindings = r_rel.get("results", {}).get("bindings", [])
                for b in bindings:
                    if "p_fwd" in b:
                        fwd_names.add(b["p_fwd"]["value"].split("/")[-1].lower())
                    if "p_rev" in b:
                        rev_names.add(b["p_rev"]["value"].split("/")[-1].lower())

            if not fwd_names and not rev_names and (s_can != subject or o_can != obj):
                s_orig_t = self._term(subject)
                o_orig_t = self._term(obj)
                q_orig = f"""
                SELECT DISTINCT ?p_fwd ?p_rev WHERE {{
                  {{ {s_orig_t} ?p_fwd {o_orig_t} . FILTER(?p_fwd != <http://dbpedia.org/ontology/wikiPageWikiLink>) }}
                  UNION
                  {{ {o_orig_t} ?p_rev {s_orig_t} . FILTER(?p_rev != <http://dbpedia.org/ontology/wikiPageWikiLink>) }}
                }} LIMIT 30
                """
                r_orig = self.query_json(q_orig, timeout=5.0)
                if r_orig is not None:
                    responded = True
                    for b in r_orig.get("results", {}).get("bindings", []):
                        if "p_fwd" in b:
                            fwd_names.add(b["p_fwd"]["value"].split("/")[-1].lower())
                        if "p_rev" in b:
                            rev_names.add(b["p_rev"]["value"].split("/")[-1].lower())

        all_rel_names = fwd_names | rev_names
        targets = ALIAS_MAP.get(pred_name, {pred_name})
        has_target_rel = bool(all_rel_names & targets) or any(pred_name in n for n in all_rel_names)

        conflicts = CONFLICT_MAP.get(pred_name, set())
        has_conflicting = (
            bool(all_rel_names & conflicts) and not has_target_rel
        ) or (
            pred_name in ["deathplace", "birthplace"]
            and bool(all_rel_names & {"birthplace", "deathplace"} - targets)
        )
        has_other_rel = bool(all_rel_names) and not has_target_rel and not has_conflicting

        direct_match = bool(fwd_names & {pred_name})
        property_match = bool(fwd_names & targets)
        redirect_match = has_target_rel and (s_can != subject or o_can != obj) and not direct_match
        inverse_match = bool(rev_names & targets)

        # Degrees & Predicate counts (only if needed)
        subj_out = subj_in = obj_in = obj_out = 0
        if not all_rel_names:
            subj_out, subj_in = self.get_entity_degrees(s_can) if s_uri else (0, 0)
            obj_in, obj_out = self.get_entity_degrees(o_can) if o_uri else (0, 0)
        pred_subj, pred_obj = self.get_predicate_counts(predicate) if p_uri else (0, 0)

        # Shared neighbors & 2-hop paths: only queried if no direct relation was found
        shared = two_hops = 0
        if not all_rel_names and s_uri and o_uri and subject != obj:
            s_t = self._term(s_can)
            p_t = self._term(predicate)
            o_t = self._term(o_can)
            r_2h = self.query_json(
                f"SELECT (COUNT(*) AS ?hops) WHERE {{ "
                f"{{ {s_t} ?p1 ?mid . ?mid {p_t} {o_t} . }} "
                f"UNION {{ {s_t} {p_t} ?mid . ?mid ?p2 {o_t} . }} }} LIMIT 50",
                timeout=2.0,
            )
            if r_2h is not None:
                responded = True
                two_hops = self._int_val(r_2h, "hops")

            r_sh = self.query_json(
                f"SELECT (COUNT(DISTINCT ?mid) AS ?shared) WHERE {{ "
                f"{{ {s_t} ?p1 ?mid . ?mid ?p2 {o_t} . }} "
                f"UNION {{ {s_t} ?p3 ?mid . {o_t} ?p4 ?mid . }} }} LIMIT 50",
                timeout=2.0,
            )
            if r_sh is not None:
                responded = True
                shared = self._int_val(r_sh, "shared")

        # Types & Type compatibility (only if needed)
        s_types = self.get_entity_types(s_can) if (s_uri and not all_rel_names) else set()
        o_types = self.get_entity_types(o_can) if (o_uri and not all_rel_names) else set()
        expected_ranges = self.get_predicate_ranges(predicate) if (p_uri and not all_rel_names) else set()
        type_compat = (
            bool(expected_ranges & o_types)
            if (expected_ranges and o_types)
            else bool(s_types and o_types)
        )

        signals = KGSignals(
            direct_match=direct_match,
            property_match=property_match,
            redirect_match=redirect_match,
            inverse_match=inverse_match,
            has_target_relation=has_target_rel,
            has_conflicting_relation=has_conflicting,
            has_other_relation=has_other_rel,
            fwd_rel_count=len(fwd_names),
            rev_rel_count=len(rev_names),
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
            offline_or_empty=not (
                responded
                or direct_match
                or property_match
                or has_target_rel
                or subj_out > 0
                or obj_in > 0
            ),
        )
        if responded:
            self.cache.put(self.endpoint, key, signals.__dict__)
        return signals
