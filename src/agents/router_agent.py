import logging
from typing import Any

from src.agents.agent_schema import AgentTool, AgentType, RouterInput, RouterPlan
from src.agents.agent_utils import (
    extract_first_json_object,
    normalize_aggregation_method,
    normalize_period_token,
)
from src.agents.helper_agent import HelperAgent
from src.agents.llm_base import LLMBase
from src.agents.query_cache import QueryCache
from src.prompts.router_prompts import ROUTER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class RouterAgent(LLMBase):
    """Plan and dispatch user questions to the appropriate analytics tool."""

    _cache: QueryCache | None = None

    @classmethod
    def _get_cache(cls, config_path: str = "config/llm_config.conf") -> QueryCache:
        """Return the shared class-level cache, creating it on first access."""
        if cls._cache is None:
            cls._cache = QueryCache.from_config(config_path)
        return cls._cache

    def __init__(self, llm_config_path: str = "config/llm_config.conf") -> None:
        super().__init__(
            llm_config_path=llm_config_path,
            config_section="router",
            default_temperature=0.0,
            default_top_p=1.0,
            default_max_tokens=350,
        )
        self._model = self.init_chat_model(
            model_name=self.model_name,
            allow_test_key=True,
        )

    def ask(
        self,
        question: str,
        validated_rows: list[dict[str, Any]],
        chat_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Route a user question and return a final assistant response payload.

        Args:
            question: The user prompt to analyze and route.
            validated_rows: Normalized transaction rows available for analysis.
            chat_history: Optional recent conversation history used for context.

        Returns:
            A response payload that includes answer text, routing metadata,
            and tool execution details.
        """
        cache = self._get_cache()
        data_hash = QueryCache.compute_data_hash(validated_rows)

        plan = self.plan_with_schema(
            RouterInput(question=question, chat_history=chat_history)
        )

        if plan.needs_clarification:
            clarification = (
                plan.clarification_question.strip()
                or "Could you clarify which period or category you want to analyze?"
            )
            return {
                "answer": clarification,
                "rowsScanned": len(validated_rows),
                "toolUsed": False,
                "confidence": plan.confidence,
                "route": plan.route.value,
                "toolName": plan.tool_name.value,
                "toolParams": plan.tool_params,
                "needsClarification": True,
            }

        # Check cache AFTER routing — keyed on tool + params + data, not query text.
        cached = cache.get(
            plan.tool_name.value, plan.tool_params, data_hash, user_query=question
        )
        if cached is not None:
            return cached

        helper = HelperAgent()
        result = helper.ask_with_routed_tool(
            question=question,
            validated_rows=validated_rows,
            tool_name=plan.tool_name.value,
            tool_params=plan.tool_params,
            chat_history=chat_history,
        )
        result["route"] = plan.route.value
        result["toolName"] = plan.tool_name.value
        result["toolParams"] = plan.tool_params
        result["needsClarification"] = False
        result["confidence"] = plan.confidence

        cache.put(plan.tool_name.value, plan.tool_params, result, data_hash)

        return result

    def plan_with_schema(self, payload: RouterInput) -> RouterPlan:
        """
        Return a validated route plan for a user question.

        Args:
            payload: Structured router input containing question and chat history.

        Returns:
            A normalized and validated RouterPlan.
        """
        raw_text = self._invoke_router_model(
            question=payload.question,
            chat_history=payload.chat_history,
        )

        parsed = extract_first_json_object(raw_text)
        if not isinstance(parsed, dict):
            logger.info(
                "[Router] query=%r | parsed=None (fallback)",
                payload.question,
            )
            return self._fallback_plan()

        logger.info(
            "[Router] query=%r | parsed=%r",
            payload.question,
            parsed,
        )
        return self._normalize_plan(parsed)

    def _invoke_router_model(
        self,
        question: str,
        chat_history: list[dict[str, Any]] | None,
    ) -> str:
        """
        Invoke the router model and return plain text output.

        Args:
            question: The user question to route.
            chat_history: Optional chat context to improve routing decisions.

        Returns:
            The raw text content returned by the model.
        """
        history_lines = self.history_lines(chat_history, limit=8)
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Recent chat context:\n{chr(10).join(history_lines) if history_lines else '(none)'}\n\n"
                    "Return JSON only."
                ),
            },
        ]

        try:
            response = self._model.invoke(messages)
            return self._content_to_text(getattr(response, "content", "")).strip()
        except Exception:
            logger.exception("router_agent model invocation failed")
            return ""

    def _normalize_plan(self, parsed: dict[str, Any]) -> RouterPlan:
        """
        Coerce model output to supported routes and tool parameter contracts.

        Args:
            parsed: Untrusted model-produced plan dictionary.

        Returns:
            A RouterPlan constrained to supported routes and params.
        """
        route = AgentType.from_value(
            parsed.get("route", AgentType.HELPER.value),
            default=AgentType.HELPER,
        )
        if route is not AgentType.HELPER:
            route = AgentType.HELPER

        tool_name = AgentTool.from_value(
            parsed.get("tool_name", AgentTool.SPENDING_BREAKDOWN.value),
            default=AgentTool.SPENDING_BREAKDOWN,
        )

        tool_params = parsed.get("tool_params", {})
        if not isinstance(tool_params, dict):
            tool_params = {}

        needs_clarification = bool(parsed.get("needs_clarification", False))
        clarification_question = str(
            parsed.get("clarification_question", "") or ""
        ).strip()

        confidence = str(parsed.get("confidence", "high") or "high").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"

        normalized_params = self._normalize_tool_params(
            tool_name=tool_name,
            tool_params=tool_params,
        )

        # If a comparison is requested without both periods, force a follow-up
        # clarification so the downstream tool call remains well-formed.
        if not needs_clarification and self._missing_required_params(
            tool_name, normalized_params
        ):
            needs_clarification = True
            if not clarification_question:
                clarification_question = "Do you want a comparison between two periods, or a single-period total for this month?"

        # Pie charts can't represent a two-period comparison — ask for bar instead.
        if (
            not needs_clarification
            and tool_name is AgentTool.COMPARE_SPENDING_PERIODS
            and normalized_params.get("chart_type") == "pie"
        ):
            needs_clarification = True
            clarification_question = (
                "Pie charts can't show a comparison between two periods. "
                "Would you like a bar graph instead?"
            )

        return RouterPlan(
            route=route,
            tool_name=tool_name,
            tool_params=normalized_params,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            confidence=confidence,
        )

    @staticmethod
    def _normalize_tool_params(
        tool_name: AgentTool,
        tool_params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalize tool params to a strict contract before execution.

        Args:
            tool_name: Selected helper tool identifier.
            tool_params: Raw model-produced parameter mapping.

        Returns:
            A normalized parameter dictionary that matches tool expectations.
        """

        def normalize_period(value: Any, default: str) -> Any:
            if isinstance(value, dict):
                return value
            return normalize_period_token(value or default)

        if tool_name is AgentTool.COMPARE_SPENDING_PERIODS:
            raw_chart_type = (
                str(tool_params.get("chart_type", "") or "").strip().lower()
            )
            if raw_chart_type == "pie":
                compare_chart_type = "pie"
            else:
                # For comparisons, any bar variant maps to grouped_bar
                # so both periods are shown side-by-side.
                compare_chart_type = "grouped_bar"
            return {
                "period_1": normalize_period(
                    tool_params.get("period_1", "this_month"),
                    default="this_month",
                ),
                "period_2": normalize_period(
                    tool_params.get("period_2", "last_month"),
                    default="last_month",
                ),
                "category": str(tool_params.get("category", "") or "").strip(),
                "aggregation_method": normalize_aggregation_method(
                    str(tool_params.get("aggregation_method", "sum") or "sum")
                ),
                "weekly_average": bool(tool_params.get("weekly_average", False)),
                "include_chart": bool(tool_params.get("include_chart", False)),
                "chart_type": compare_chart_type,
            }

        period_value = tool_params.get("period")
        normalized_period = normalize_period(period_value, default="this_month")

        return {
            "category": str(tool_params.get("category", "") or "").strip(),
            "this_month": bool(tool_params.get("this_month", False)),
            "period": normalized_period if period_value is not None else None,
            "aggregation_method": normalize_aggregation_method(
                str(tool_params.get("aggregation_method", "sum") or "sum")
            ),
            "top_n": max(0, int(tool_params.get("top_n", 0) or 0)),
            "include_chart": bool(tool_params.get("include_chart", False)),
            "chart_type": (
                "pie"
                if str(tool_params.get("chart_type", "") or "").strip().lower() == "pie"
                else "bar"
            ),
        }

    @staticmethod
    def _missing_required_params(
        tool_name: AgentTool,
        tool_params: dict[str, Any],
    ) -> bool:
        """
        Check whether required params are absent for the selected tool.

        Args:
            tool_name: Selected helper tool identifier.
            tool_params: Normalized parameter mapping for that tool.

        Returns:
            True when required fields are missing, otherwise False.
        """
        if tool_name is AgentTool.COMPARE_SPENDING_PERIODS:
            period_1 = tool_params.get("period_1")
            period_2 = tool_params.get("period_2")
            period_1_missing = not period_1
            period_2_missing = not period_2
            return period_1_missing or period_2_missing

        return False

    @staticmethod
    def _fallback_plan() -> RouterPlan:
        """
        Return a safe fallback route when parsing fails.

        Returns:
            A conservative RouterPlan that asks the user to clarify intent.
        """
        return RouterPlan(
            route=AgentType.HELPER,
            tool_name=AgentTool.SPENDING_BREAKDOWN,
            tool_params={
                "category": "",
                "this_month": False,
                "aggregation_method": "sum",
                "top_n": 0,
            },
            needs_clarification=True,
            clarification_question=(
                "I can help with spending analysis. Do you want a total for this month, or a comparison with another period?"
            ),
            confidence="low",
        )
