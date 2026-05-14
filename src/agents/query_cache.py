import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any

from pyhocon import ConfigFactory

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIZE = 100


class QueryCache:
    """In-memory LRU cache keyed on (tool_name, tool_params, data_hash).

    After the RouterAgent classifies a query, the cache checks whether the
    same tool + params + data combination has already been executed.  This
    avoids duplicate tool execution while allowing the router to run every
    time (cheap) and never confusing queries that route to different tools.

    Cache key dimensions:
    - tool_name: prevents collisions across different tools
    - tool_params: captures structured router output deterministically
    - data_hash: invalidates cache when validated transaction data changes
    """

    def __init__(self, max_size: int = _DEFAULT_MAX_SIZE) -> None:
        """Initialize cache storage.

        Args:
            max_size: Maximum number of entries to retain. Values smaller
                than 1 are coerced to 1.
        """
        self._max_size = max(1, max_size)
        # OrderedDict for LRU eviction — most-recently-used entries move to end.
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @property
    def size(self) -> int:
        """Return the number of entries currently in the cache."""
        return len(self._store)

    @staticmethod
    def _make_key(tool_name: str, tool_params: dict[str, Any], data_hash: str) -> str:
        """Build a deterministic cache key from router output and data state.

        Args:
            tool_name: Name of the selected tool.
            tool_params: Structured arguments passed to the tool.
            data_hash: Hash fingerprint for current validated rows.

        Returns:
            SHA-256 hex digest used as the internal cache key.
        """
        # Sort keys so semantically identical dicts produce the same key.
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
        """Return cached response for a tool invocation, if present.

        Args:
            tool_name: Name of the selected tool.
            tool_params: Tool arguments from routing output.
            data_hash: Hash of validated row data for staleness safety.
            user_query: Original user text used only for observability logs.

        Returns:
            Cached response payload when found, otherwise ``None``.
        """
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

        # Mark as recently used for LRU semantics.
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
        """Store a tool-execution result in cache, deduplicating on key.

        If the same key already exists, the call is a no-op.

        Args:
            tool_name: Name of the selected tool.
            tool_params: Tool arguments from routing output.
            response: Response payload to cache.
            data_hash: Hash of validated row data for staleness safety.
        """
        key = self._make_key(tool_name, tool_params, data_hash)

        # Keep first-write behavior to avoid accidental mutation churn.
        if key in self._store:
            return

        self._store[key] = response
        # Newly inserted entries are most-recently-used.
        self._store.move_to_end(key)

        # LRU eviction
        while len(self._store) > self._max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("cache EVICT key=%r", evicted_key)

    def clear(self) -> None:
        """Remove all cache entries."""
        self._store.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_data_hash(validated_rows: list[dict[str, Any]]) -> str:
        """Compute a lightweight hash over validated rows for staleness detection.

        Builds a deterministic canonical representation of row content so
        semantically meaningful edits produce a new hash fingerprint.

        Args:
            validated_rows: Validated transaction records.

        Returns:
            Short hex digest used to detect data changes between queries.
        """
        if not validated_rows:
            return "empty"

        canonical_rows: list[dict[str, Any]] = []
        for row in validated_rows:
            if not isinstance(row, dict):
                continue

            canonical_row: dict[str, Any] = {}
            for key in sorted(row.keys()):
                value = row.get(key)
                if isinstance(value, (int, float)):
                    canonical_row[key] = round(float(value), 6)
                elif value is None:
                    canonical_row[key] = ""
                else:
                    canonical_row[key] = str(value).strip()
            canonical_rows.append(canonical_row)

        canonical_rows.sort(
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
        )
        raw = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @classmethod
    def from_config(
        cls,
        config_path: str = "config/llm_config.conf",
    ) -> "QueryCache":
        """Build a QueryCache from the cache section in config.

        Args:
            config_path: Path to HOCON configuration file.

        Returns:
            QueryCache instance initialized with configured max_size,
            or defaults when parsing fails.
        """
        try:
            cfg = ConfigFactory.parse_file(config_path)
            max_size = int(cfg.get("cache.max_size", _DEFAULT_MAX_SIZE))
        except Exception:
            # Fail open with safe defaults so cache never blocks app startup.
            max_size = _DEFAULT_MAX_SIZE

        return cls(max_size=max_size)
