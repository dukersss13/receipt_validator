import copy

import pytest

from src.agents.query_cache import CacheEntry, QueryCache

# ── fixtures ──────────────────────────────────────────────────────────

SAMPLE_RESPONSE = {
    "answer": "You spent $120 on food.",
    "rowsScanned": 5,
    "toolUsed": True,
    "confidence": "high",
    "route": "helper_agent",
    "toolName": "spending_breakdown",
    "toolParams": {"category": "food"},
    "needsClarification": False,
}

SAMPLE_CHART_RESPONSE = {
    **SAMPLE_RESPONSE,
    "chart": {
        "type": "bar",
        "title": "Spending breakdown by category",
        "currency": "USD",
        "labels": ["Food", "Travel"],
        "values": [120.0, 80.0],
    },
}

DATA_HASH_A = "abc123"
DATA_HASH_B = "def456"


@pytest.fixture
def cache() -> QueryCache:
    return QueryCache(similarity_threshold=0.85, max_size=5)


# ── exact match ───────────────────────────────────────────────────────


def test_exact_hit(cache: QueryCache) -> None:
    cache.put("How much did I spend on food?", SAMPLE_RESPONSE, DATA_HASH_A)
    result = cache.get("How much did I spend on food?", DATA_HASH_A)
    assert result is not None
    assert result["answer"] == SAMPLE_RESPONSE["answer"]


def test_exact_hit_case_insensitive(cache: QueryCache) -> None:
    cache.put("How much did I spend on food?", SAMPLE_RESPONSE, DATA_HASH_A)
    result = cache.get("HOW MUCH DID I SPEND ON FOOD?", DATA_HASH_A)
    assert result is not None


def test_exact_hit_ignores_punctuation(cache: QueryCache) -> None:
    cache.put("How much did I spend on food?", SAMPLE_RESPONSE, DATA_HASH_A)
    result = cache.get("How much did I spend on food", DATA_HASH_A)
    assert result is not None


# ── fuzzy match ───────────────────────────────────────────────────────


def test_fuzzy_hit(cache: QueryCache) -> None:
    cache.put("How much did I spend on food?", SAMPLE_RESPONSE, DATA_HASH_A)
    result = cache.get("How much did I spend on foods?", DATA_HASH_A)
    assert result is not None
    assert result["answer"] == SAMPLE_RESPONSE["answer"]


def test_fuzzy_miss_below_threshold(cache: QueryCache) -> None:
    cache.put("How much did I spend on food?", SAMPLE_RESPONSE, DATA_HASH_A)
    result = cache.get("Compare this month vs last month", DATA_HASH_A)
    assert result is None


# ── chart response caching ────────────────────────────────────────────


def test_chart_response_cached(cache: QueryCache) -> None:
    cache.put("Show me a bar chart", SAMPLE_CHART_RESPONSE, DATA_HASH_A)
    result = cache.get("Show me a bar chart", DATA_HASH_A)
    assert result is not None
    assert result.get("chart") is not None
    assert result["chart"]["type"] == "bar"
    assert result["chart"]["labels"] == ["Food", "Travel"]


# ── data hash invalidation ───────────────────────────────────────────


def test_stale_data_hash_misses(cache: QueryCache) -> None:
    cache.put("How much did I spend on food?", SAMPLE_RESPONSE, DATA_HASH_A)
    result = cache.get("How much did I spend on food?", DATA_HASH_B)
    assert result is None


def test_refreshed_data_hash_overwrites(cache: QueryCache) -> None:
    cache.put("How much did I spend on food?", SAMPLE_RESPONSE, DATA_HASH_A)
    new_response = {**SAMPLE_RESPONSE, "answer": "You spent $200 on food."}
    cache.put("How much did I spend on food?", new_response, DATA_HASH_B)
    result = cache.get("How much did I spend on food?", DATA_HASH_B)
    assert result is not None
    assert result["answer"] == "You spent $200 on food."


# ── deduplication ─────────────────────────────────────────────────────


def test_duplicate_put_is_noop(cache: QueryCache) -> None:
    cache.put("How much did I spend on food?", SAMPLE_RESPONSE, DATA_HASH_A)
    cache.put("How much did I spend on food?", SAMPLE_RESPONSE, DATA_HASH_A)
    assert cache.size == 1


def test_put_does_not_overwrite_same_hash(cache: QueryCache) -> None:
    original = copy.deepcopy(SAMPLE_RESPONSE)
    cache.put("How much did I spend on food?", original, DATA_HASH_A)
    modified = {**SAMPLE_RESPONSE, "answer": "Different answer"}
    cache.put("How much did I spend on food?", modified, DATA_HASH_A)
    result = cache.get("How much did I spend on food?", DATA_HASH_A)
    # First insert wins — same hash means no overwrite.
    assert result["answer"] == original["answer"]


# ── LRU eviction ──────────────────────────────────────────────────────


def test_lru_eviction(cache: QueryCache) -> None:
    # Use diverse queries so fuzzy matching won't cross-match them.
    diverse = [
        "total food spend",
        "compare months travel",
        "show pie chart groceries",
        "average transport cost weekly",
        "top five categories overall",
        "breakdown entertainment expenses",
    ]
    for i, q in enumerate(diverse):
        cache.put(q, {"answer": f"answer {i}"}, DATA_HASH_A)
    # max_size=5, so the first entry should have been evicted.
    assert cache.size == 5
    assert cache.get(diverse[0], DATA_HASH_A) is None
    assert cache.get(diverse[5], DATA_HASH_A) is not None


def test_lru_access_refreshes_position(cache: QueryCache) -> None:
    diverse = [
        "total food spend",
        "compare months travel",
        "show pie chart groceries",
        "average transport cost weekly",
        "top five categories overall",
        "breakdown entertainment expenses",
    ]
    for i in range(5):
        cache.put(diverse[i], {"answer": f"answer {i}"}, DATA_HASH_A)
    # Access first entry to refresh its position.
    cache.get(diverse[0], DATA_HASH_A)
    # Insert one more to trigger eviction — diverse[1] should be evicted, not diverse[0].
    cache.put(diverse[5], {"answer": "answer 5"}, DATA_HASH_A)
    assert cache.get(diverse[0], DATA_HASH_A) is not None
    assert cache.get(diverse[1], DATA_HASH_A) is None


# ── empty / edge cases ────────────────────────────────────────────────


def test_empty_query_returns_none(cache: QueryCache) -> None:
    assert cache.get("", DATA_HASH_A) is None


def test_put_empty_query_is_noop(cache: QueryCache) -> None:
    cache.put("", SAMPLE_RESPONSE, DATA_HASH_A)
    assert cache.size == 0


def test_clear(cache: QueryCache) -> None:
    cache.put("query 1", SAMPLE_RESPONSE, DATA_HASH_A)
    cache.put("query 2", SAMPLE_RESPONSE, DATA_HASH_A)
    cache.clear()
    assert cache.size == 0


# ── compute_data_hash ─────────────────────────────────────────────────


def test_data_hash_empty_rows() -> None:
    assert QueryCache.compute_data_hash([]) == "empty"


def test_data_hash_deterministic() -> None:
    rows = [{"Transaction Total": 10.0}, {"Transaction Total": 20.5}]
    h1 = QueryCache.compute_data_hash(rows)
    h2 = QueryCache.compute_data_hash(rows)
    assert h1 == h2


def test_data_hash_changes_with_data() -> None:
    rows_a = [{"Transaction Total": 10.0}]
    rows_b = [{"Transaction Total": 10.0}, {"Transaction Total": 5.0}]
    assert QueryCache.compute_data_hash(rows_a) != QueryCache.compute_data_hash(rows_b)


# ── from_config ───────────────────────────────────────────────────────


def test_from_config_defaults() -> None:
    # Non-existent path should fall back to defaults without error.
    c = QueryCache.from_config("nonexistent.conf")
    assert c._threshold == 0.85
    assert c._max_size == 100
