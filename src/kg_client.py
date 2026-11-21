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
