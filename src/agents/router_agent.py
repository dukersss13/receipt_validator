import logging
from typing import Any
from enum import Enum

from src.agents.agent_schema import Tools, RouterInput, RouterPlan
from src.agents.agent_utils import (
    extract_first_json_object,
    normalize_aggregation_method,
    normalize_period_token,
)
from src.agents.agent_tools import AgentTools
from src.agents.llm_base import LLMBase
from src.agents.query_cache import QueryCache
from src.prompts.router_prompts import ROUTER_ANSWER_PROMPT, ROUTER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Suggestions(Enum):
    FOOD_SPENDING = "how much did i spend on food?"
    MOST_CATEGORY = "what's my top spending category?"
    TOP_5 = "show my top 5 categories"
    SPENDING_CHART = "chart my spending"


class RouterAgent(LLMBase):
    """Plan and dispatch user questions to the appropriate analytics tool."""

    _cache: QueryCache | None = None
    _MAX_HISTORY: int = 10

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
        self.tools = AgentTools()
        self._chat_history: list[dict[str, Any]] = []
        self._pending_plan: RouterPlan | None = None

    @staticmethod
    def check_suggestions(question: str) -> tuple[bool, str | None]:
        """
        Check if the question matches any of the predefined suggestions.
        """
        question_lower = question.lower()

        for suggestion in Suggestions:
            if question_lower == suggestion.value:
                return True, suggestion.name

        return False, None

    @staticmethod
    def _build_suggestion_plan(suggestion_name: str) -> RouterPlan:
        """
        Build a router plan for a suggested action.
        """
        if suggestion_name == Suggestions.FOOD_SPENDING.name:
            plan = RouterPlan(
                tool_name=Tools.SPENDING_BREAKDOWN,
                tool_params={
                    "category": "food",
                    "this_month": False,
                    "period": None,
                    "aggregation_method": "sum",
                    "top_n": 0,
                    "include_chart": False,
                    "chart_type": "bar",
                },
                needs_clarification=False,
                confidence="high",
            )
        elif suggestion_name == Suggestions.MOST_CATEGORY.name:
            plan = RouterPlan(
                tool_name=Tools.SPENDING_BREAKDOWN,
                tool_params={
                    "category": "",
                    "this_month": False,
                    "period": None,
                    "aggregation_method": "sum",
                    "top_n": 1,
                    "include_chart": False,
                    "chart_type": "bar",
                },
                needs_clarification=False,
                confidence="high",
            )
        elif suggestion_name == Suggestions.TOP_5.name:
            plan = RouterPlan(
                tool_name=Tools.SPENDING_BREAKDOWN,
                tool_params={
                    "category": "",
                    "this_month": False,
                    "period": None,
                    "aggregation_method": "sum",
                    "top_n": 5,
                    "include_chart": False,
                    "chart_type": "bar",
                },
                needs_clarification=False,
                confidence="high",
            )
        elif suggestion_name == Suggestions.SPENDING_CHART.name:
            plan = RouterPlan(
                tool_name=Tools.SPENDING_BREAKDOWN,
                tool_params={
                    "category": "",
                    "this_month": False,
                    "period": None,
                    "aggregation_method": "sum",
                    "top_n": 0,
                    "include_chart": True,
                    "chart_type": "bar",
                },
                needs_clarification=False,
                confidence="high",
            )

        return plan

    def ask(
        self,
        question: str,
        validated_rows: list[dict[str, Any]],
        chat_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Route a user question and return a final assistant response payload.

        If a prior request triggered a clarification and stored a pending plan,
        the user's follow-up is merged into that plan instead of re-routing.
        Maintains an internal chat history (last 10 turns) for routing context.

        Args:
            question: The user prompt to analyze and route.
            validated_rows: Normalized transaction rows available for analysis.
            chat_history: Optional external chat history; merged with internal.

        Returns:
            A response payload that includes answer text, routing metadata,
            and tool execution details.
        """
        # Merge external history into internal on first call or when supplied.
        if chat_history:
            self._merge_external_history(chat_history)

        cache = self._get_cache()
        data_hash = QueryCache.compute_data_hash(validated_rows)

        suggestion_check, suggestion_name = RouterAgent.check_suggestions(question)
        # If we have a pending plan from a prior clarification, resolve it.
        if self._pending_plan is not None:
            plan = self._resolve_pending_plan(question)

        elif suggestion_check and suggestion_name is not None:
            # Deterministic shortcut: chart all validated transactions.
            # Since this is a suggested action, we can take it at face value without needing clarification.
            plan = RouterAgent._build_suggestion_plan(suggestion_name)

        else:
            plan = self.plan_with_schema(
                RouterInput(
                    question=question,
                    chat_history=self._chat_history,
                )
            )

        if plan.needs_clarification:
            self._pending_plan = plan
            clarification = (
                plan.clarification_question.strip()
                or "Could you clarify which period or category you want to analyze?"
            )
            quick_replies = self._build_quick_replies(plan)
            result = {
                "answer": clarification,
                "rowsScanned": len(validated_rows),
                "toolUsed": False,
                "confidence": plan.confidence,
                "toolName": plan.tool_name.value, # type: ignore
                "toolParams": plan.tool_params,
                "needsClarification": True,
                "quickReplies": quick_replies,
            }
            self._append_history(question, clarification)
            return result

        # Clear any pending plan — this request is fully resolved.
        self._pending_plan = None

        # Check cache AFTER routing — keyed on tool + params + data, not query text.
        cached = cache.get(
            plan.tool_name.value, plan.tool_params, data_hash, user_query=question # type: ignore
        )
        if cached is not None:
            self._append_history(question, cached.get("answer", ""))
            return cached

        self.tools.set_validated_rows(validated_rows)

        tool_output = self.tools.execute_tool(
            selected_tool=plan.tool_name, # type: ignore
            tool_params=plan.tool_params,
        )
    
        result: dict[str, Any] = {
            "answer": self._render_answer(question, tool_output),
            "rowsScanned": len(validated_rows),
            "toolUsed": True,
        }

        if isinstance(tool_output, dict):
            suggestions = tool_output.get("timeframe_suggestions")
            if isinstance(suggestions, list) and suggestions:
                result["quickReplies"] = [str(item) for item in suggestions[:3]]

        if isinstance(tool_output, dict) and tool_output.get("chart") is not None:
            result["chart"] = tool_output.get("chart")

        if isinstance(tool_output, dict) and tool_output.get("top_categories"):
            result["top_categories"] = tool_output["top_categories"]

        result["toolName"] = plan.tool_name.value # type: ignore
        result["toolParams"] = plan.tool_params
        result["needsClarification"] = False
        result["confidence"] = plan.confidence

        cache.put(plan.tool_name.value, plan.tool_params, result, data_hash) # type: ignore
        self._append_history(question, result.get("answer", ""))

        return result

    def _render_answer(self, question: str, tool_output: dict[str, Any]) -> str:
        """
        Render a natural language answer from the structured tool output.
        """
        messages = [
            {"role": "system", "content": ROUTER_ANSWER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Tool Output:\n{tool_output}\n\n"
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

    def _append_history(self, question: str, answer: str) -> None:
        """
        Append a user/assistant turn and cap at ``_MAX_HISTORY`` entries.
        This ensures that the conversation history doesn't grow indefinitely.
        """
        self._chat_history.append({"role": "user", "text": question})
        self._chat_history.append({"role": "assistant", "text": answer})

        if len(self._chat_history) > self._MAX_HISTORY * 2:
            self._chat_history = self._chat_history[-(self._MAX_HISTORY * 2) :]

    def _merge_external_history(self, external: list[dict[str, Any]]) -> None:
        """
        Seed internal history from external source if internal is empty.
        """
        if self._chat_history:
            return
        for turn in external[-(self._MAX_HISTORY * 2) :]:
            if isinstance(turn, dict) and "role" in turn and "text" in turn:
                self._chat_history.append({"role": turn["role"], "text": turn["text"]})

    def _resolve_pending_plan(self, user_answer: str) -> RouterPlan:
        """Merge a user's clarification answer into the pending plan.

        Args:
            user_answer: The follow-up text from the user.

        Returns:
            An updated RouterPlan with the missing fields filled in.
        """
        plan = self._pending_plan
        assert plan is not None
        self._pending_plan = None

        answer = user_answer.strip().lower()

        # Case 1: Pie chart on comparison — user confirms bar graph.
        if (
            plan.tool_name is Tools.COMPARE_SPENDING_PERIODS
            and plan.tool_params.get("chart_type") == "pie"
        ):
            if "bar" in answer or "yes" in answer or "sure" in answer:
                plan.tool_params["chart_type"] = "grouped_bar"
                plan.needs_clarification = False
                plan.clarification_question = ""
                return plan
            # User declined — remove chart entirely.
            plan.tool_params["include_chart"] = False
            plan.tool_params.pop("chart_type", None)
            plan.needs_clarification = False
            plan.clarification_question = ""
            return plan

        # Case 2: Missing periods for comparison, or fallback ambiguity.
        # Re-route through the LLM with the pending plan as context so it
        # only needs to fill in the missing fields.
        merged_plan = self._reroute_with_context(user_answer, plan)
        return merged_plan

    def _reroute_with_context(
        self, question: str, prior_plan: RouterPlan
    ) -> RouterPlan:
        """Re-invoke the router with the prior partial plan as context.

        Args:
            question: The user's clarification follow-up.
            prior_plan: The incomplete plan from the prior turn.

        Returns:
            A new RouterPlan that inherits unresolved params from the prior plan.
        """
        new_plan = self.plan_with_schema(
            RouterInput(question=question, chat_history=self._chat_history)
        )

        # Carry forward params the user didn't re-specify.
        merged_params = {**prior_plan.tool_params}
        for key, value in new_plan.tool_params.items():
            if value not in (None, "", 0, False):
                merged_params[key] = value
        new_plan.tool_params = merged_params

        # If the new plan kept the same tool but added the missing fields,
        # it's resolved. Otherwise trust the new plan's clarification state.
        if (
            new_plan.tool_name == prior_plan.tool_name
            and not self._missing_required_params(
                new_plan.tool_name, new_plan.tool_params # type: ignore
            )
        ):
            new_plan.needs_clarification = False
            new_plan.clarification_question = ""

        return new_plan

    @staticmethod
    def _build_quick_replies(plan: RouterPlan) -> list[str]:
        """Generate contextual quick-reply options for a clarification.

        Args:
            plan: The plan that triggered the clarification.

        Returns:
            A list of suggested reply strings the UI can render as buttons.
        """
        # Pie chart not supported for comparison.
        if (
            plan.tool_name is Tools.COMPARE_SPENDING_PERIODS
            and plan.tool_params.get("chart_type") == "pie"
        ):
            return ["Yes, use a bar graph", "No chart"]

        # Missing periods — suggest common comparisons.
        if plan.tool_name is Tools.COMPARE_SPENDING_PERIODS:
            return [
                "This month vs last month",
                "This month vs 2 months ago",
                "Past 3 months vs prior 3 months",
            ]

        # Fallback / ambiguous single-period request.
        return [
            "Total spending this month",
            "Compare this month vs last month",
            "Top 5 categories",
        ]

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

        parsed: dict | None = extract_first_json_object(raw_text)

        logger.info(
            "[Router] query=%r | parsed=%r",
            payload.question,
            parsed,
        )
        return self.extract_router_plan(parsed)

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

    def extract_router_plan(self, parsed: dict | None) -> RouterPlan:
        """Extract and normalize a RouterPlan from model output.

        Args:
            parsed: Untrusted model-produced plan dictionary containing
                tool_name, tool_params, needs_clarification,
                clarification_question, and confidence fields.

        Returns:
            A validated RouterPlan constrained to supported tools and params.
        """
        if not parsed:
            return RouterPlan(
                tool_name=None,
                tool_params={},
                needs_clarification=True,
                clarification_question="I'm sorry, I didn't understand that. Can you please rephrase your question?",
                confidence="low",
            )

        tool_name = parsed.get("tool_name", None)
        if tool_name == Tools.SPENDING_BREAKDOWN.value:
            tool_name_enum = Tools.SPENDING_BREAKDOWN
        elif tool_name == Tools.COMPARE_SPENDING_PERIODS.value:
            tool_name_enum = Tools.COMPARE_SPENDING_PERIODS

        raw_params = parsed.get("tool_params", {})
        if not isinstance(raw_params, dict):
            raw_params = {}
        normalized_params = self._normalize_tool_params(tool_name=tool_name_enum, 
                                                        tool_params=raw_params)

        needs_clarification = bool(parsed.get("needs_clarification", False))
        clarification_question = str(parsed.get("clarification_question", ""))

        confidence = str(parsed.get("confidence", "high"))
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"

        if not needs_clarification and self._missing_required_params(
            tool_name_enum, normalized_params
        ):
            needs_clarification = True
            clarification_question = clarification_question or (
                "Do you want a comparison between two periods, "
                "or a single-period total for this month?"
            )

        if (
            not needs_clarification
            and tool_name_enum is Tools.COMPARE_SPENDING_PERIODS
            and normalized_params.get("chart_type") == "pie"
        ):
            needs_clarification = True
            clarification_question = (
                "Pie charts can't show a comparison between two periods. "
                "Would you like a bar graph instead?"
            )

        return RouterPlan(
            tool_name=tool_name_enum,
            tool_params=normalized_params,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            confidence=confidence,
        )

    @staticmethod
    def _normalize_tool_params(
        tool_name: Tools | None,
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

        if tool_name is Tools.COMPARE_SPENDING_PERIODS:
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
            "category": str(tool_params.get("category", "")),
            "this_month": bool(tool_params.get("this_month", False)),
            "period": normalized_period if period_value is not None else None,
            "aggregation_method": normalize_aggregation_method(
                str(tool_params.get("aggregation_method", "sum") or "sum")
            ),
            "top_n": max(0, int(tool_params.get("top_n", 0))),
            "include_chart": bool(tool_params.get("include_chart", False)),
            "chart_type": (
                "pie"
                if str(tool_params.get("chart_type", "")).strip().lower() == "pie"
                else "bar"
            ),
        }

    @staticmethod
    def _missing_required_params(
        tool_name: Tools,
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
        if tool_name is Tools.COMPARE_SPENDING_PERIODS:
            period_1 = tool_params.get("period_1")
            period_2 = tool_params.get("period_2")
            period_1_missing = not period_1
            period_2_missing = not period_2
            return period_1_missing or period_2_missing

        return False
