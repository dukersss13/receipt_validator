import hashlib
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from pyhocon import ConfigFactory

logger = logging.getLogger(__name__)

_DEFAULT_SIMILARITY_THRESHOLD = 0.85
_DEFAULT_MAX_SIZE = 100


@dataclass(slots=True)
class CacheEntry:
    """Single cached query→response pair."""

    query: str
    normalized_query: str
    response: dict[str, Any]
    data_hash: str
    created_at: float = field(default_factory=time.time)


class QueryCache:
    """In-memory query→response cache with exact and fuzzy matching.

    Supports O(1) exact lookups on normalized query text and a linear-scan
    fuzzy fallback using ``difflib.SequenceMatcher``.  Entries are invalidated
    when the underlying data changes (detected via ``data_hash``).
    """

    def __init__(
        self,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        self._threshold = max(0.0, min(1.0, similarity_threshold))
        self._max_size = max(1, max_size)
        # OrderedDict for LRU eviction — most-recently-used entries move to end.
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    @property
    def size(self) -> int:
        return len(self._store)

    def get(self, query: str, data_hash: str) -> dict[str, Any] | None:
        """Return a cached response for *query*, or ``None`` on miss.

        Checks exact normalized match first (O(1)), then falls back to a
        linear scan for fuzzy matches above the similarity threshold.
        Entries with a stale ``data_hash`` are skipped.
        """
        normalized = self._normalize(query)
        if not normalized:
            return None

        # --- exact match ---
        entry = self._store.get(normalized)
        if entry is not None and entry.data_hash == data_hash:
            self._store.move_to_end(normalized)
            logger.info("cache HIT (exact) query=%r", query)
            return entry.response

        # --- fuzzy match ---
        best_key: str | None = None
        best_ratio: float = 0.0
        for key, entry in self._store.items():
            if entry.data_hash != data_hash:
                continue
            ratio = SequenceMatcher(None, normalized, entry.normalized_query).ratio()
            if ratio >= self._threshold and ratio > best_ratio:
                best_ratio = ratio
                best_key = key

        if best_key is not None:
            self._store.move_to_end(best_key)
            logger.info(
                "cache HIT (fuzzy %.2f) query=%r matched=%r",
                best_ratio,
                query,
                self._store[best_key].query,
            )
            return self._store[best_key].response

        logger.debug("cache MISS query=%r", query)
        return None

    def put(
        self,
        query: str,
        response: dict[str, Any],
        data_hash: str,
    ) -> None:
        """Store a query→response pair, deduplicating on normalized query.

        If the normalized query already exists with the same ``data_hash``,
        the call is a no-op (no duplicate entries).  If it exists with a
        *different* ``data_hash`` the entry is refreshed.
        """
        normalized = self._normalize(query)
        if not normalized:
            return

        existing = self._store.get(normalized)
        if existing is not None and existing.data_hash == data_hash:
            # Already cached with same data — skip.
            return

        entry = CacheEntry(
            query=query,
            normalized_query=normalized,
            response=response,
            data_hash=data_hash,
        )

        # Remove old key first so reinsertion lands at the end (most-recent).
        if normalized in self._store:
            del self._store[normalized]

        self._store[normalized] = entry
        self._store.move_to_end(normalized)

        # LRU eviction
        while len(self._store) > self._max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("cache EVICT key=%r", evicted_key)

    def clear(self) -> None:
        """Drop all cached entries."""
        self._store.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(query: str) -> str:
        """Lowercase, strip punctuation, and collapse whitespace."""
        text = str(query or "").strip().lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def compute_data_hash(validated_rows: list[dict[str, Any]]) -> str:
        """Compute a lightweight hash over validated rows for staleness detection.

        Uses row count and the sum of transaction totals so that any data
        change (add, remove, edit) produces a different hash.
        """
        if not validated_rows:
            return "empty"

        row_count = len(validated_rows)
        total_sum = 0.0
        for row in validated_rows:
            try:
                total_sum += float(row.get("Transaction Total", 0) or 0)
            except (TypeError, ValueError):
                pass

        raw = f"{row_count}:{total_sum:.4f}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @classmethod
    def from_config(
        cls,
        config_path: str = "config/llm_config.conf",
    ) -> "QueryCache":
        """Build a ``QueryCache`` from the ``cache`` section of the LLM config."""
        try:
            cfg = ConfigFactory.parse_file(config_path)
            threshold = float(
                cfg.get("cache.similarity_threshold", _DEFAULT_SIMILARITY_THRESHOLD)
            )
            max_size = int(cfg.get("cache.max_size", _DEFAULT_MAX_SIZE))
        except Exception:
            threshold = _DEFAULT_SIMILARITY_THRESHOLD
            max_size = _DEFAULT_MAX_SIZE

        return cls(similarity_threshold=threshold, max_size=max_size)
