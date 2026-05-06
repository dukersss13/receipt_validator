import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any

from pyhocon import ConfigFactory

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIZE = 100


class QueryCache:
    """In-memory cache keyed on (tool_name, tool_params, data_hash).

    After the RouterAgent classifies a query, the cache checks whether the
    same tool + params + data combination has already been executed.  This
    avoids duplicate HelperAgent calls while allowing the router to run every
    time (cheap) and never confusing queries that route to different tools.
    """

    def __init__(self, max_size: int = _DEFAULT_MAX_SIZE) -> None:
        self._max_size = max(1, max_size)
        # OrderedDict for LRU eviction — most-recently-used entries move to end.
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @property
    def size(self) -> int:
        return len(self._store)

    @staticmethod
    def _make_key(tool_name: str, tool_params: dict[str, Any], data_hash: str) -> str:
        """Build a deterministic cache key from tool routing output."""
        params_json = json.dumps(tool_params, sort_keys=True, default=str)
        raw = f"{tool_name}:{params_json}:{data_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        data_hash: str,
        user_query: str = "",
    ) -> dict[str, Any] | None:
        """Return a cached response for this tool+params+data, or ``None``."""
        key = self._make_key(tool_name, tool_params, data_hash)
        entry = self._store.get(key)
        if entry is None:
            logger.info(
                "[Cache] MISS | query=%r | tool=%s | params=%r | cached_response=None",
                user_query,
                tool_name,
                tool_params,
            )
            return None

        self._store.move_to_end(key)
        logger.info(
            "[Cache] HIT | query=%r | tool=%s | params=%r | cached_response=%r",
            user_query,
            tool_name,
            tool_params,
            entry.get("answer", "")[:120],
        )
        return entry

    def put(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        response: dict[str, Any],
        data_hash: str,
    ) -> None:
        """Store a tool-execution result, deduplicating on key.

        If the same key already exists the call is a no-op.
        """
        key = self._make_key(tool_name, tool_params, data_hash)

        if key in self._store:
            return

        self._store[key] = response
        self._store.move_to_end(key)

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
            max_size = int(cfg.get("cache.max_size", _DEFAULT_MAX_SIZE))
        except Exception:
            max_size = _DEFAULT_MAX_SIZE

        return cls(max_size=max_size)
