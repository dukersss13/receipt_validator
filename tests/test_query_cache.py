import copy

import pytest

from src.agents.query_cache import QueryCache

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

TOOL_SPENDING = "spending_breakdown"
TOOL_COMPARE = "compare_spending_periods"

PARAMS_FOOD = {"category": "food", "this_month": True, "aggregation_method": "sum"}
PARAMS_FOOD_CHART = {**PARAMS_FOOD, "include_chart": True, "chart_type": "bar"}
PARAMS_COMPARE = {"period_1": "this_month", "period_2": "last_month", "category": ""}


@pytest.fixture
def cache() -> QueryCache:
    return QueryCache(max_size=5)


# ── exact hit ─────────────────────────────────────────────────────────


def test_exact_hit(cache: QueryCache) -> None:
    cache.put(TOOL_SPENDING, PARAMS_FOOD, SAMPLE_RESPONSE, DATA_HASH_A)
    result = cache.get(TOOL_SPENDING, PARAMS_FOOD, DATA_HASH_A)
    assert result is not None
    assert result["answer"] == SAMPLE_RESPONSE["answer"]


def test_same_tool_different_params_miss(cache: QueryCache) -> None:
    cache.put(TOOL_SPENDING, PARAMS_FOOD, SAMPLE_RESPONSE, DATA_HASH_A)
    result = cache.get(TOOL_SPENDING, {"category": "travel"}, DATA_HASH_A)
    assert result is None


def test_different_tool_same_params_miss(cache: QueryCache) -> None:
    cache.put(TOOL_SPENDING, PARAMS_FOOD, SAMPLE_RESPONSE, DATA_HASH_A)
    result = cache.get(TOOL_COMPARE, PARAMS_FOOD, DATA_HASH_A)
    assert result is None


# ── the key scenario: same topic, different tools ─────────────────────


def test_spending_vs_compare_no_false_positive(cache: QueryCache) -> None:
    """'spendings this month' and 'this month vs last month' route to
    different tools, so they must never collide in the cache."""
    cache.put(TOOL_SPENDING, PARAMS_FOOD, SAMPLE_RESPONSE, DATA_HASH_A)
    result = cache.get(TOOL_COMPARE, PARAMS_COMPARE, DATA_HASH_A)
    assert result is None


# ── chart response caching ────────────────────────────────────────────


def test_chart_response_cached(cache: QueryCache) -> None:
    cache.put(TOOL_SPENDING, PARAMS_FOOD_CHART, SAMPLE_CHART_RESPONSE, DATA_HASH_A)
    result = cache.get(TOOL_SPENDING, PARAMS_FOOD_CHART, DATA_HASH_A)
    assert result is not None
    assert result.get("chart") is not None
    assert result["chart"]["type"] == "bar"
    assert result["chart"]["labels"] == ["Food", "Travel"]


# ── data hash invalidation ───────────────────────────────────────────


def test_stale_data_hash_misses(cache: QueryCache) -> None:
    cache.put(TOOL_SPENDING, PARAMS_FOOD, SAMPLE_RESPONSE, DATA_HASH_A)
    result = cache.get(TOOL_SPENDING, PARAMS_FOOD, DATA_HASH_B)
    assert result is None


def test_new_data_hash_creates_separate_entry(cache: QueryCache) -> None:
    cache.put(TOOL_SPENDING, PARAMS_FOOD, SAMPLE_RESPONSE, DATA_HASH_A)
    new_response = {**SAMPLE_RESPONSE, "answer": "You spent $200 on food."}
    cache.put(TOOL_SPENDING, PARAMS_FOOD, new_response, DATA_HASH_B)
    result = cache.get(TOOL_SPENDING, PARAMS_FOOD, DATA_HASH_B)
    assert result is not None
    assert result["answer"] == "You spent $200 on food."


# ── deduplication ─────────────────────────────────────────────────────


def test_duplicate_put_is_noop(cache: QueryCache) -> None:
    cache.put(TOOL_SPENDING, PARAMS_FOOD, SAMPLE_RESPONSE, DATA_HASH_A)
    cache.put(TOOL_SPENDING, PARAMS_FOOD, SAMPLE_RESPONSE, DATA_HASH_A)
    assert cache.size == 1


def test_put_does_not_overwrite_same_key(cache: QueryCache) -> None:
    original = copy.deepcopy(SAMPLE_RESPONSE)
    cache.put(TOOL_SPENDING, PARAMS_FOOD, original, DATA_HASH_A)
    modified = {**SAMPLE_RESPONSE, "answer": "Different answer"}
    cache.put(TOOL_SPENDING, PARAMS_FOOD, modified, DATA_HASH_A)
    result = cache.get(TOOL_SPENDING, PARAMS_FOOD, DATA_HASH_A)
    assert result["answer"] == original["answer"]


# ── LRU eviction ──────────────────────────────────────────────────────


def test_lru_eviction(cache: QueryCache) -> None:
    params_list = [{"category": f"cat_{i}", "this_month": True} for i in range(6)]
    for i, p in enumerate(params_list):
        cache.put(TOOL_SPENDING, p, {"answer": f"answer {i}"}, DATA_HASH_A)
    # max_size=5, so the first entry should have been evicted.
    assert cache.size == 5
    assert cache.get(TOOL_SPENDING, params_list[0], DATA_HASH_A) is None
    assert cache.get(TOOL_SPENDING, params_list[5], DATA_HASH_A) is not None


def test_lru_access_refreshes_position(cache: QueryCache) -> None:
    params_list = [{"category": f"cat_{i}", "this_month": True} for i in range(6)]
    for i in range(5):
        cache.put(TOOL_SPENDING, params_list[i], {"answer": f"answer {i}"}, DATA_HASH_A)
    # Access first entry to refresh its position.
    cache.get(TOOL_SPENDING, params_list[0], DATA_HASH_A)
    # Insert one more — params_list[1] should be evicted, not params_list[0].
    cache.put(TOOL_SPENDING, params_list[5], {"answer": "answer 5"}, DATA_HASH_A)
    assert cache.get(TOOL_SPENDING, params_list[0], DATA_HASH_A) is not None
    assert cache.get(TOOL_SPENDING, params_list[1], DATA_HASH_A) is None


# ── edge cases ────────────────────────────────────────────────────────


def test_empty_params(cache: QueryCache) -> None:
    cache.put(TOOL_SPENDING, {}, SAMPLE_RESPONSE, DATA_HASH_A)
    result = cache.get(TOOL_SPENDING, {}, DATA_HASH_A)
    assert result is not None


def test_clear(cache: QueryCache) -> None:
    cache.put(TOOL_SPENDING, PARAMS_FOOD, SAMPLE_RESPONSE, DATA_HASH_A)
    cache.put(TOOL_COMPARE, PARAMS_COMPARE, SAMPLE_RESPONSE, DATA_HASH_A)
    cache.clear()
    assert cache.size == 0


# ── _make_key determinism ─────────────────────────────────────────────


def test_make_key_deterministic() -> None:
    k1 = QueryCache._make_key(TOOL_SPENDING, PARAMS_FOOD, DATA_HASH_A)
    k2 = QueryCache._make_key(TOOL_SPENDING, PARAMS_FOOD, DATA_HASH_A)
    assert k1 == k2


def test_make_key_param_order_irrelevant() -> None:
    p1 = {"category": "food", "this_month": True}
    p2 = {"this_month": True, "category": "food"}
    assert QueryCache._make_key(TOOL_SPENDING, p1, DATA_HASH_A) == QueryCache._make_key(
        TOOL_SPENDING, p2, DATA_HASH_A
    )


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


def test_data_hash_changes_when_row_content_changes_with_same_totals() -> None:
    rows_a = [
        {
            "Transaction Business Name": "Store A",
            "Transaction Total": 10.0,
            "Transaction Category": "Food",
        },
        {
            "Transaction Business Name": "Store B",
            "Transaction Total": 20.0,
            "Transaction Category": "Travel",
        },
    ]
    rows_b = [
        {
            "Transaction Business Name": "Store A",
            "Transaction Total": 10.0,
            "Transaction Category": "Utilities",
        },
        {
            "Transaction Business Name": "Store B",
            "Transaction Total": 20.0,
            "Transaction Category": "Travel",
        },
    ]
    assert QueryCache.compute_data_hash(rows_a) != QueryCache.compute_data_hash(rows_b)


def test_data_hash_order_independent_for_same_rows() -> None:
    rows_a = [
        {"Transaction Business Name": "Store A", "Transaction Total": 10.0},
        {"Transaction Business Name": "Store B", "Transaction Total": 20.0},
    ]
    rows_b = [
        {"Transaction Business Name": "Store B", "Transaction Total": 20.0},
        {"Transaction Business Name": "Store A", "Transaction Total": 10.0},
    ]
    assert QueryCache.compute_data_hash(rows_a) == QueryCache.compute_data_hash(rows_b)


# ── from_config ───────────────────────────────────────────────────────


def test_from_config_defaults() -> None:
    c = QueryCache.from_config("nonexistent.conf")
    assert c._max_size == 100
