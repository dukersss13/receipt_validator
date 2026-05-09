from typing import Any

from src.agents.agent_schema import RouterInput
from src.agents.router_agent import RouterAgent


def test_router_plan_normalizes_spending_params(monkeypatch: Any) -> None:
    router = RouterAgent()

    monkeypatch.setattr(
        router,
        "_invoke_router_model",
        lambda question, chat_history: (
            '{"route":"helper_agent","tool_name":"spending_breakdown",'
            '"tool_params":{"category":"Food","this_month":true,'
            '"aggregation_method":"mean","top_n":3},'
            '"needs_clarification":false,"clarification_question":"",'
            '"confidence":"high"}'
        ),
    )

    plan = router.plan_with_schema(payload=router_input("Top 3 food categories"))

    assert plan.tool_name == "spending_breakdown"
    assert plan.tool_params["category"] == "Food"
    assert plan.tool_params["this_month"] is True
    assert plan.tool_params["aggregation_method"] == "average"
    assert plan.tool_params["top_n"] == 3
    assert plan.tool_params["include_chart"] is False
    assert plan.tool_params["chart_type"] == "bar"
    assert plan.needs_clarification is False


def test_router_plan_normalizes_spending_chart_intent(monkeypatch: Any) -> None:
    router = RouterAgent()

    monkeypatch.setattr(
        router,
        "_invoke_router_model",
        lambda question, chat_history: (
            '{"route":"helper_agent","tool_name":"spending_breakdown",'
            '"tool_params":{"category":"","this_month":false,'
            '"aggregation_method":"sum","top_n":0,'
            '"include_chart":true,"chart_type":"pie"},'
            '"needs_clarification":false,"clarification_question":"",'
            '"confidence":"high"}'
        ),
    )

    plan = router.plan_with_schema(
        payload=router_input("Create a pie chart of my spending")
    )

    assert plan.tool_name == "spending_breakdown"
    assert plan.tool_params["include_chart"] is True
    assert plan.tool_params["chart_type"] == "pie"


def test_router_plan_normalizes_compare_params(monkeypatch: Any) -> None:
    router = RouterAgent()

    monkeypatch.setattr(
        router,
        "_invoke_router_model",
        lambda question, chat_history: (
            '{"route":"helper_agent","tool_name":"compare_spending_periods",'
            '"tool_params":{"period_1":"2026-04","period_2":"last_month",'
            '"category":"travel","aggregation_method":"total",'
            '"weekly_average":true},"needs_clarification":false,'
            '"clarification_question":"","confidence":"medium"}'
        ),
    )

    plan = router.plan_with_schema(payload=router_input("Compare travel spending"))

    assert plan.tool_name == "compare_spending_periods"
    assert plan.tool_params["period_1"] == "2026-04"
    assert plan.tool_params["period_2"] == "last_month"
    assert plan.tool_params["aggregation_method"] == "sum"
    assert plan.tool_params["weekly_average"] is True
    assert plan.tool_params["chart_type"] == "grouped_bar"
    assert plan.confidence == "medium"


def test_router_keeps_structured_period_specs(monkeypatch: Any) -> None:
    router = RouterAgent()

    monkeypatch.setattr(
        router,
        "_invoke_router_model",
        lambda question, chat_history: (
            '{"route":"helper_agent","tool_name":"compare_spending_periods",'
            '"tool_params":{"period_1":{"kind":"segment","position":"first","unit":"month","n":3},'
            '"period_2":{"kind":"segment","position":"last","unit":"month","n":3},'
            '"category":"","aggregation_method":"sum","weekly_average":false,'
            '"include_chart":true},"needs_clarification":false,'
            '"clarification_question":"","confidence":"high"}'
        ),
    )

    plan = router.plan_with_schema(
        payload=router_input("Compare first 3 months vs last 3 months with a chart")
    )

    assert plan.tool_name == "compare_spending_periods"
    assert isinstance(plan.tool_params["period_1"], dict)
    assert plan.tool_params["period_1"]["kind"] == "segment"
    assert isinstance(plan.tool_params["period_2"], dict)
    assert plan.tool_params["period_2"]["position"] == "last"
    assert plan.tool_params["include_chart"] is True


def test_router_accepts_spending_period_override(monkeypatch: Any) -> None:
    router = RouterAgent()

    monkeypatch.setattr(
        router,
        "_invoke_router_model",
        lambda question, chat_history: (
            '{"route":"helper_agent","tool_name":"spending_breakdown",'
            '"tool_params":{"category":"food","this_month":false,'
            '"period":{"kind":"relative_window","unit":"month","n":2},'
            '"aggregation_method":"sum","top_n":0},'
            '"needs_clarification":false,"clarification_question":"",'
            '"confidence":"high"}'
        ),
    )

    plan = router.plan_with_schema(
        payload=router_input("How much did I spend on food in the past 2 months?")
    )

    assert plan.tool_name == "spending_breakdown"
    assert isinstance(plan.tool_params["period"], dict)
    assert plan.tool_params["period"]["kind"] == "relative_window"


def test_router_asks_clarification_when_model_requests_it(monkeypatch: Any) -> None:
    router = RouterAgent()

    monkeypatch.setattr(
        router,
        "_invoke_router_model",
        lambda question, chat_history: (
            '{"route":"helper_agent","tool_name":"spending_breakdown",'
            '"tool_params":{},"needs_clarification":true,'
            '"clarification_question":"Which period do you want to analyze?",'
            '"confidence":"low"}'
        ),
    )

    result = router.ask(
        question="Show my spending",
        validated_rows=[],
        chat_history=None,
    )

    assert result["toolUsed"] is False
    assert result["needsClarification"] is True
    assert "period" in result["answer"].lower()


def test_router_dispatches_routed_tool(monkeypatch: Any) -> None:
    router = RouterAgent()

    monkeypatch.setattr(
        router,
        "_invoke_router_model",
        lambda question, chat_history: (
            '{"route":"helper_agent","tool_name":"spending_breakdown",'
            '"tool_params":{"category":"food","this_month":true,'
            '"aggregation_method":"sum","top_n":0},'
            '"needs_clarification":false,"clarification_question":"",'
            '"confidence":"high"}'
        ),
    )

    monkeypatch.setattr(
        "src.agents.agent_tools.AgentTools.execute_tool",
        lambda self, tool_name, tool_params: {
            "status": "ok",
            "type": "total",
            "value": 10.0,
        },
    )
    monkeypatch.setattr(
        "src.agents.agent_tools.AgentTools.render_answer",
        lambda tool_name, tool_output: "You spent $10.00 on food this month.",
    )

    rows = [
        {
            "Transaction Business Name": "Cafe",
            "Transaction Total": 10.0,
            "Transaction Date": "2026-05-01",
            "Transaction Category": "Food",
        }
    ]
    result = router.ask("How much did I spend on food this month?", rows)

    assert result["toolUsed"] is True
    assert result["toolName"] == "spending_breakdown"
    assert result["toolParams"]["category"] == "food"
    assert result["rowsScanned"] == 1


def router_input(question: str) -> Any:
    return RouterInput(question=question, chat_history=None)


def test_router_normalizes_chart_with_period(monkeypatch: Any) -> None:
    router = RouterAgent()

    monkeypatch.setattr(
        router,
        "_invoke_router_model",
        lambda question, chat_history: (
            '{"route":"helper_agent","tool_name":"spending_breakdown",'
            '"tool_params":{"category":"","this_month":false,'
            '"period":{"kind":"relative_window","unit":"month","n":2},'
            '"aggregation_method":"sum","top_n":0,'
            '"include_chart":true,"chart_type":"bar"},'
            '"needs_clarification":false,"clarification_question":"",'
            '"confidence":"high"}'
        ),
    )

    plan = router.plan_with_schema(
        payload=router_input(
            "Give me a chart for all category spendings the last 2 months"
        )
    )

    assert plan.tool_name == "spending_breakdown"
    assert plan.tool_params["include_chart"] is True
    assert plan.tool_params["chart_type"] == "bar"
    assert isinstance(plan.tool_params["period"], dict)
    assert plan.tool_params["period"]["kind"] == "relative_window"
    assert plan.tool_params["period"]["n"] == 2
    assert plan.tool_params["this_month"] is False
