from src.agents.agent_tools import AgentTools


def test_spending_breakdown_returns_bar_chart_payload() -> None:
    tools = AgentTools(
        validated_rows=[
            {
                "Transaction Business Name": "Cafe",
                "Transaction Total": 14.5,
                "Transaction Date": "2026-05-01",
                "Transaction Category": "Food",
            },
            {
                "Transaction Business Name": "Grocer",
                "Transaction Total": 40.0,
                "Transaction Date": "2026-05-02",
                "Transaction Category": "Grocery",
            },
        ]
    )

    payload = tools.execute_spending_breakdown(
        this_month=False,
        include_chart=True,
        chart_type="bar",
    )

    chart = payload.get("chart")
    assert isinstance(chart, dict)
    assert chart.get("type") == "bar"
    assert chart.get("labels")
    assert chart.get("values")


def test_spending_breakdown_returns_pie_chart_payload() -> None:
    tools = AgentTools(
        validated_rows=[
            {
                "Transaction Business Name": "Ride",
                "Transaction Total": 21.0,
                "Transaction Date": "2026-05-03",
                "Transaction Category": "Transport",
            },
            {
                "Transaction Business Name": "Market",
                "Transaction Total": 55.0,
                "Transaction Date": "2026-05-04",
                "Transaction Category": "Grocery",
            },
        ]
    )

    payload = tools.execute_spending_breakdown(
        this_month=False,
        include_chart=True,
        chart_type="pie",
    )

    chart = payload.get("chart")
    assert isinstance(chart, dict)
    assert chart.get("type") == "pie"
    assert chart.get("labels")
    assert chart.get("values")


def test_spending_breakdown_chart_with_period() -> None:
    tools = AgentTools(
        validated_rows=[
            {
                "Transaction Business Name": "Cafe",
                "Transaction Total": 12.0,
                "Transaction Date": "2026-04-10",
                "Transaction Category": "Food",
            },
            {
                "Transaction Business Name": "Gas",
                "Transaction Total": 45.0,
                "Transaction Date": "2026-04-15",
                "Transaction Category": "Transport",
            },
            {
                "Transaction Business Name": "Grocer",
                "Transaction Total": 60.0,
                "Transaction Date": "2026-05-02",
                "Transaction Category": "Grocery",
            },
        ]
    )

    payload = tools.execute_spending_breakdown(
        this_month=False,
        period={"kind": "relative_window", "unit": "month", "n": 2},
        include_chart=True,
        chart_type="bar",
    )

    assert payload["status"] == "ok"
    chart = payload.get("chart")
    assert isinstance(chart, dict)
    assert chart["type"] == "bar"
    assert len(chart["labels"]) > 0
    assert len(chart["values"]) == len(chart["labels"])
    assert chart.get("currency") == "USD"


def test_compare_spending_periods_chart_payload() -> None:
    tools = AgentTools(
        validated_rows=[
            {
                "Transaction Business Name": "Cafe",
                "Transaction Total": 20.0,
                "Transaction Date": "2026-05-01",
                "Transaction Category": "Food",
            },
            {
                "Transaction Business Name": "Market",
                "Transaction Total": 35.0,
                "Transaction Date": "2026-05-03",
                "Transaction Category": "Grocery",
            },
            {
                "Transaction Business Name": "Diner",
                "Transaction Total": 15.0,
                "Transaction Date": "2026-04-10",
                "Transaction Category": "Food",
            },
            {
                "Transaction Business Name": "Shop",
                "Transaction Total": 42.0,
                "Transaction Date": "2026-04-12",
                "Transaction Category": "Grocery",
            },
        ]
    )

    payload = tools.execute_compare_spending_periods(
        period_1="this_month",
        period_2="last_month",
        include_chart=True,
    )

    assert payload["status"] == "ok"
    assert payload["type"] == "comparison"
    chart = payload.get("chart")
    assert isinstance(chart, dict)
    assert chart["type"] == "grouped_bar"
    assert len(chart["x"]) > 0
    assert len(chart["series"]) == 2
    assert chart["series"][0]["name"]
    assert chart["series"][1]["name"]
    assert len(chart["series"][0]["values"]) == len(chart["x"])
    assert chart.get("currency") == "USD"


def test_spending_no_results_with_timeframe_returns_suggestions() -> None:
    tools = AgentTools(
        validated_rows=[
            {
                "Transaction Business Name": "Cafe",
                "Transaction Total": 14.5,
                "Transaction Date": "2020-01-01",
                "Transaction Category": "Food",
            },
            {
                "Transaction Business Name": "Grocer",
                "Transaction Total": 40.0,
                "Transaction Date": "2020-01-02",
                "Transaction Category": "Grocery",
            },
        ]
    )

    payload = tools.execute_spending_breakdown(
        this_month=True,
        include_chart=False,
    )

    assert payload["status"] == "no_results"
    assert payload["timeframe_requested"] is True
    assert isinstance(payload.get("timeframe_suggestions"), list)
    assert payload.get("timeframe_suggestions")

    answer = AgentTools.render_answer("spending_breakdown", payload)
    assert "could not find any transactions" in answer.lower()


def test_spending_total_payload_does_not_include_top_categories() -> None:
    tools = AgentTools(
        validated_rows=[
            {
                "Transaction Business Name": "Cafe",
                "Transaction Total": 14.5,
                "Transaction Date": "2026-05-01",
                "Transaction Category": "Food",
            },
            {
                "Transaction Business Name": "Grocer",
                "Transaction Total": 40.0,
                "Transaction Date": "2026-05-02",
                "Transaction Category": "Grocery",
            },
        ]
    )

    payload = tools.execute_spending_breakdown(
        category="Food",
        this_month=False,
        top_n=0,
        include_chart=False,
    )

    assert payload["status"] == "ok"
    assert payload["type"] == "total"
    assert "top_categories" not in payload
