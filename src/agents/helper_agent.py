import re
import json
import logging
from datetime import date, timedelta
from typing import Any, Iterator

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import BaseTool, tool
from src.agents.agent_schema import AgentInput, AgentOutput, AgentTool
from src.agents.agent_utils import (
    category_matches,
    get_category_fuzzy_min_ratio,
    normalize_aggregation_method,
    normalize_period_token,
)
from src.agents.llm_base import LLMBase
from src.prompts.arvee_prompts import ARVEE_ANSWER_PROMPT, ARVEE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MONTH_NAME_TO_NUMBER: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


class HelperAgent(LLMBase):
    """
    Answer natural-language questions over validated transaction rows.

    Wraps a LangChain tool-calling agent backed by Gemini. Exposes both a
    synchronous ``ask`` method and a streaming ``stream_answer`` generator.
    """

    def __init__(
        self,
        llm_config_path: str = "config/llm_config.conf",
    ) -> None:
        """
        Build the tool-calling agent used by the chat endpoint.

        Args:
            llm_config_path: Path to the HOCON config file that controls model
                selection and sampling parameters.
        """
        super().__init__(
            llm_config_path=llm_config_path,
            config_section="helper_agent",
            default_temperature=0.1,
            default_top_p=1.0,
            default_max_tokens=500,
        )

        self._model = self.init_chat_model(
            model_name=self.model_name,
            allow_test_key=True,
        )
        self._category_fuzzy_min_ratio = get_category_fuzzy_min_ratio()
        # Will be replaced on each ask/stream_answer call with fresh row data
        self._validated_rows: list[dict[str, Any]] = []
        # Init HelperAgent + tools
        self._agent = create_agent(
            model=self._model,
            tools=[
                self._breakdown_spending_tool(),
                self._compare_spending_periods_tool(),
            ],
            system_prompt=ARVEE_SYSTEM_PROMPT,
            name="Arvee",
        )

    def _breakdown_spending_tool(self) -> BaseTool:
        """
        Build and return the ``spending_breakdown`` tool function for the agent.

        The inner function is registered as a LangChain tool and is called by the
        agent for category-specific totals, averages, and top-N ranked categories.
        Supported aggregations are ``sum`` and ``average``.

        Returns:
            A LangChain tool that the agent invokes to analyse ``self._validated_rows``.
        """

        @tool
        def spending_breakdown(
            category: str = "",
            this_month: bool = False,
            aggregation_method: str = "sum",
            top_n: int = 0,
        ) -> str:
            """
            Use this tool for spending analytics over validated transactions.

            Call this tool when the user asks for:
            - total spend
            - average spend
            - category-filtered spend
            - top-N categories by spend or average spend

            Default behavior:
            - if no timeframe is specified, set ``this_month`` to False to include all transactions
            - only set ``this_month`` to True when the user explicitly says "this month"
            - leave ``category`` empty to include all categories
            - use ``aggregation_method`` as ``sum`` unless average is requested
            - set ``top_n`` only when a ranked top-N result is requested

            Args:
                category: Optional spending category to filter by (e.g. 'food', 'travel', 'grocery').
                          Leave empty to include all categories.
                this_month: If True, only include transactions from the current month.
                            If False, include all available transactions.
                aggregation_method: Aggregation to apply: 'sum' (total) or 'average'.
                top_n: If greater than 0, return the top N categories by the selected aggregation.

            Returns:
                A text summary of the spending analysis.
            """
            payload = self.execute_spending_breakdown(
                category=category,
                this_month=this_month,
                aggregation_method=aggregation_method,
                top_n=top_n,
            )
            return json.dumps(payload)

        return spending_breakdown

    def _compare_spending_periods_tool(self) -> BaseTool:
        """
        Build and return the ``compare_spending_periods`` tool for the agent.

        Returns:
            A LangChain tool that compares two user-requested periods.
        """

        @tool
        def compare_spending_periods(
            period_1: str = "this_month",
            period_2: str = "last_month",
            category: str = "",
            aggregation_method: str = "sum",
            weekly_average: bool = False,
        ) -> str:
            """
            Use this tool to compare spending between two periods.

            Call this tool when the user asks for:
            - "this month vs last month" spending
            - "this month vs N months ago" spending
            - changes/increase/decrease between two periods
            - average spent per week comparisons (set ``weekly_average`` to True)

            Period formats supported:
            - this_month
            - last_month
            - N_months_ago (for example: 2_months_ago)
            - YYYY-MM (for example: 2026-04)

            Args:
                period_1: First comparison period token.
                period_2: Second comparison period token.
                category: Optional category filter applied to both periods.
                aggregation_method: "sum" or "average".
                weekly_average: If True, compare average spend per week instead of period-level values.

            Returns:
                A text summary showing both period values and change.
            """
            payload = self.execute_compare_spending_periods(
                period_1=period_1,
                period_2=period_2,
                category=category,
                aggregation_method=aggregation_method,
                weekly_average=weekly_average,
            )
            return json.dumps(payload)

        return compare_spending_periods

    def execute_spending_breakdown(
        self,
        category: str = "",
        this_month: bool = True,
        aggregation_method: str = "sum",
        top_n: int = 0,
        period: Any = None,
        include_chart: bool = False,
        chart_type: str = "bar",
    ) -> dict[str, Any]:
        """Execute the spending_breakdown computation on current validated rows."""
        frame = HelperAgent._to_frame(self._validated_rows)
        if frame.empty:
            return {"status": "no_data"}

        scoped = frame.copy()
        if period is not None:
            start, end, _ = HelperAgent._resolve_period(period, frame=frame)
            scoped = HelperAgent._slice_period(scoped, start, end, category="")
        elif this_month:
            now = pd.Timestamp(date.today())
            scoped = scoped[
                (scoped["Transaction Date"].dt.year == now.year)
                & (scoped["Transaction Date"].dt.month == now.month)
            ]

        category = str(category or "").strip()
        if category:
            scoped = scoped[
                scoped["Transaction Category"]
                .astype(str)
                .map(
                    lambda candidate: category_matches(
                        category,
                        candidate,
                        min_ratio=self._category_fuzzy_min_ratio,
                    )
                )
            ]

        if scoped.empty:
            return {
                "status": "no_results",
                "category_filter": category or None,
                "this_month": bool(this_month),
                "period": period,
            }

        method = normalize_aggregation_method(aggregation_method)
        top_n_value = max(0, int(top_n or 0))

        # Always compute per-category breakdown for the top-5 table.
        all_grouped = (
            scoped.groupby("Transaction Category", dropna=False)["Transaction Total"]
            .agg("mean" if method == "average" else "sum")
            .sort_values(ascending=False)
        )
        top_5 = all_grouped.head(5)
        top_categories_table = [
            {"category": str(cat), "value": round(float(val), 2)}
            for cat, val in top_5.items()
        ]

        if top_n_value > 0:
            top = all_grouped.head(max(1, top_n_value))
            payload = {
                "status": "ok",
                "type": "top_categories",
                "aggregation_method": method,
                "this_month": bool(this_month),
                "period": period,
                "category_filter": category or None,
                "top_n": int(max(1, top_n_value)),
                "results": [
                    {"category": str(cat), "value": round(float(val), 2)}
                    for cat, val in top.items()
                ],
                "top_categories": top_categories_table,
            }
            if include_chart:
                payload["chart"] = HelperAgent._build_spending_breakdown_chart(
                    scoped=scoped,
                    aggregation_method=method,
                    chart_type=chart_type,
                    title_prefix="Top spending categories",
                )
            return payload

        if method == "average":
            payload = {
                "status": "ok",
                "type": "average",
                "this_month": bool(this_month),
                "period": period,
                "category_filter": category or None,
                "value": round(float(scoped["Transaction Total"].mean()), 2),
                "top_categories": top_categories_table,
            }
            if include_chart:
                payload["chart"] = HelperAgent._build_spending_breakdown_chart(
                    scoped=scoped,
                    aggregation_method=method,
                    chart_type=chart_type,
                    title_prefix="Average spend by category",
                )
            return payload

        total = float(scoped["Transaction Total"].sum())
        payload = {
            "status": "ok",
            "type": "total",
            "this_month": bool(this_month),
            "period": period,
            "category_filter": category or None,
            "value": round(total, 2),
            "top_categories": top_categories_table,
        }
        if include_chart:
            payload["chart"] = HelperAgent._build_spending_breakdown_chart(
                scoped=scoped,
                aggregation_method=method,
                chart_type=chart_type,
                title_prefix="Spending breakdown by category",
            )
        return payload

    def execute_compare_spending_periods(
        self,
        period_1: Any = "this_month",
        period_2: Any = "last_month",
        category: str = "",
        aggregation_method: str = "sum",
        weekly_average: bool = False,
        include_chart: bool = False,
    ) -> dict[str, Any]:
        """Execute the compare_spending_periods computation on current rows."""
        frame = HelperAgent._to_frame(self._validated_rows)
        if frame.empty:
            return {"status": "no_data"}

        normalized_period_1 = normalize_period_token(period_1)
        normalized_period_2 = normalize_period_token(period_2)
        start_1, end_1, label_1 = HelperAgent._resolve_period(
            normalized_period_1,
            frame=frame,
        )
        start_2, end_2, label_2 = HelperAgent._resolve_period(
            normalized_period_2,
            frame=frame,
        )

        scoped_1 = HelperAgent._slice_period(frame, start_1, end_1, category)
        scoped_2 = HelperAgent._slice_period(frame, start_2, end_2, category)

        if scoped_1.empty or scoped_2.empty:
            return {
                "status": "insufficient_data",
                "period_1": label_1,
                "period_2": label_2,
            }

        method = normalize_aggregation_method(aggregation_method)
        weekly_average = bool(weekly_average)

        value_1 = HelperAgent._aggregate_spend(
            scoped_1,
            aggregation_method=method,
            weekly_average=weekly_average,
        )
        value_2 = HelperAgent._aggregate_spend(
            scoped_2,
            aggregation_method=method,
            weekly_average=weekly_average,
        )

        delta = value_1 - value_2
        percent_change = None if value_2 == 0 else round((delta / value_2) * 100.0, 1)
        metric_label = "weekly average" if weekly_average else "value"

        return {
            "status": "ok",
            "type": "comparison",
            "metric": metric_label,
            "category_filter": category or None,
            "period_1": {
                "token": normalized_period_1,
                "label": label_1,
                "value": round(value_1, 2),
            },
            "period_2": {
                "token": normalized_period_2,
                "label": label_2,
                "value": round(value_2, 2),
            },
            "delta": round(delta, 2),
            "percent_change": percent_change,
            "chart": (
                HelperAgent._build_comparison_chart(
                    scoped_1=scoped_1,
                    scoped_2=scoped_2,
                    label_1=label_1,
                    label_2=label_2,
                    aggregation_method=method,
                )
                if bool(include_chart)
                else None
            ),
        }

    def ask_with_routed_tool(
        self,
        question: str,
        validated_rows: list[dict[str, Any]],
        tool_name: str,
        tool_params: dict[str, Any],
        chat_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Execute a specific tool with explicit params, then synthesize an answer."""
        self._validated_rows = validated_rows
        selected_tool = AgentTool.from_value(tool_name)
        params = dict(tool_params or {})

        if selected_tool is AgentTool.SPENDING_BREAKDOWN:
            tool_output = self.execute_spending_breakdown(
                category=str(params.get("category", "") or ""),
                this_month=bool(params.get("this_month", True)),
                aggregation_method=str(
                    params.get("aggregation_method", "sum") or "sum"
                ),
                top_n=int(params.get("top_n", 0) or 0),
                period=params.get("period"),
                include_chart=bool(params.get("include_chart", False)),
                chart_type=str(params.get("chart_type", "bar") or "bar"),
            )
        elif selected_tool is AgentTool.COMPARE_SPENDING_PERIODS:
            tool_output = self.execute_compare_spending_periods(
                period_1=params.get("period_1", "this_month"),
                period_2=params.get("period_2", "last_month"),
                category=str(params.get("category", "") or ""),
                aggregation_method=str(
                    params.get("aggregation_method", "sum") or "sum"
                ),
                weekly_average=bool(params.get("weekly_average", False)),
                include_chart=bool(params.get("include_chart", False)),
            )
        else:
            return self.ask(question, validated_rows, chat_history=chat_history)

        answer = self._synthesize_answer(
            question=question,
            chat_history=chat_history,
            tool_outputs=[json.dumps(tool_output)],
            fallback_text="",
        )
        response = AgentOutput(
            answer=answer,
            rowsScanned=len(validated_rows),
            toolUsed=True,
        ).to_dict()

        if isinstance(tool_output, dict) and tool_output.get("chart") is not None:
            response["chart"] = tool_output.get("chart")

        if isinstance(tool_output, dict) and tool_output.get("top_categories"):
            response["top_categories"] = tool_output["top_categories"]

        return response

    @staticmethod
    def _resolve_period(
        period_token: Any,
        frame: pd.DataFrame | None = None,
    ) -> tuple[date, date, str]:
        """Resolve a user period token into a concrete date range.

        Args:
            period_token: Period expression (for example: ``this_month``,
                ``last_month``, ``2_months_ago``, ``YYYY-MM``).

        Returns:
            Tuple of ``(start_date, end_date, human_label)``.
        """
        today = date.today()

        if isinstance(period_token, dict):
            return HelperAgent._resolve_period_spec(period_token, frame=frame)

        token = str(period_token or "this_month").strip().lower()

        if token == "this_month":
            start = date(today.year, today.month, 1)
            return start, today, "this month"

        if token == "last_month":
            first_of_this_month = date(today.year, today.month, 1)
            end = first_of_this_month - timedelta(days=1)
            start = date(end.year, end.month, 1)
            return start, end, "last month"

        past_months_match = re.match(r"^past_(\d+)_months?$", token)
        if past_months_match:
            count = max(1, int(past_months_match.group(1)))
            start = HelperAgent._first_day_of_month(
                *HelperAgent._shift_month(today.year, today.month, -(count - 1))
            )
            return start, today, f"past {count} months"

        now_aliases = {
            "now",
            "current",
            "current_month",
            "this month",
        }
        if token in now_aliases:
            start = date(today.year, today.month, 1)
            return start, today, "this month"

        months_ago_match = re.match(r"^(\d+)_months?_ago$", token)
        if months_ago_match:
            offset = int(months_ago_match.group(1))
            start, end = HelperAgent._month_range_from_offset(offset)
            return start, end, f"{offset} months ago"

        month_name_match = re.match(
            r"^(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)(?:\s+(\d{4}))?$",
            token,
        )
        if month_name_match:
            month = MONTH_NAME_TO_NUMBER[month_name_match.group(1)]
            year = (
                int(month_name_match.group(2))
                if month_name_match.group(2)
                else today.year
            )
            start = date(year, month, 1)
            end = HelperAgent._last_day_of_month(year, month)
            return start, end, f"{month_name_match.group(1).title()} {year}"

        iso_month_match = re.match(r"^(\d{4})-(\d{2})$", token)
        if iso_month_match:
            year = int(iso_month_match.group(1))
            month = int(iso_month_match.group(2))
            start = date(year, month, 1)
            if month == 12:
                next_month = date(year + 1, 1, 1)
            else:
                next_month = date(year, month + 1, 1)
            end = next_month - timedelta(days=1)
            return start, end, token

        start = date(today.year, today.month, 1)
        return start, today, "this month"

    @staticmethod
    def _resolve_period_spec(
        period_spec: dict[str, Any],
        frame: pd.DataFrame | None = None,
    ) -> tuple[date, date, str]:
        """Resolve a structured period specification to concrete start/end dates."""
        today = date.today()
        kind = str(period_spec.get("kind", "")).strip().lower()
        unit = str(period_spec.get("unit", "month") or "month").strip().lower()

        if kind in {"current", "now"}:
            if unit == "week":
                start = today - timedelta(days=today.weekday())
                return start, today, "this week"
            if unit == "day":
                return today, today, "today"
            start = date(today.year, today.month, 1)
            return start, today, "this month"

        if kind == "relative_window":
            n = max(1, int(period_spec.get("n", period_spec.get("last_n", 1)) or 1))
            if unit == "week":
                end = today
                start = end - timedelta(days=(7 * n) - 1)
                return start, end, f"past {n} weeks"
            if unit == "day":
                end = today
                start = end - timedelta(days=n - 1)
                return start, end, f"past {n} days"
            start = HelperAgent._first_day_of_month(
                *HelperAgent._shift_month(today.year, today.month, -(n - 1))
            )
            return start, today, f"past {n} months"

        if kind in {"offset", "offset_window"}:
            offset = max(
                0, int(period_spec.get("offset", period_spec.get("months_ago", 0)) or 0)
            )
            span = max(1, int(period_spec.get("span", period_spec.get("n", 1)) or 1))
            if unit == "month":
                start_month = HelperAgent._shift_month(
                    today.year, today.month, -(offset + span - 1)
                )
                end_month = HelperAgent._shift_month(today.year, today.month, -offset)
                start = HelperAgent._first_day_of_month(*start_month)
                end = HelperAgent._last_day_of_month(*end_month)
                label = (
                    f"{offset} months ago"
                    if span == 1
                    else f"{span} months ending {offset} months ago"
                )
                return start, end, label

        if kind in {"named_month", "month_name"}:
            month_raw = str(period_spec.get("month", "")).strip().lower()
            month = MONTH_NAME_TO_NUMBER.get(month_raw)
            if month:
                year = int(period_spec.get("year", today.year) or today.year)
                start = date(year, month, 1)
                end = HelperAgent._last_day_of_month(year, month)
                return start, end, f"{month_raw.title()} {year}"

        if (
            kind in {"segment", "data_segment"}
            and frame is not None
            and not frame.empty
        ):
            n = max(1, int(period_spec.get("n", 1) or 1))
            position = str(period_spec.get("position", "first")).strip().lower()
            months = sorted(
                {
                    (int(ts.year), int(ts.month))
                    for ts in frame["Transaction Date"].dropna().dt.to_pydatetime()
                }
            )
            if months:
                selected = months[:n] if position == "first" else months[-n:]
                start = HelperAgent._first_day_of_month(*selected[0])
                end = HelperAgent._last_day_of_month(*selected[-1])
                label = f"{position} {len(selected)} months"
                return start, end, label

        token = str(period_spec.get("token", "this_month") or "this_month")
        return HelperAgent._resolve_period(token, frame=frame)

    @staticmethod
    def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
        """Shift a year-month tuple by *delta* months."""
        month_index = (year * 12 + (month - 1)) + delta
        shifted_year = month_index // 12
        shifted_month = (month_index % 12) + 1
        return shifted_year, shifted_month

    @staticmethod
    def _first_day_of_month(year: int, month: int) -> date:
        """Return the first day for a given year-month."""
        return date(year, month, 1)

    @staticmethod
    def _last_day_of_month(year: int, month: int) -> date:
        """Return the last day for a given year-month."""
        if month == 12:
            return date(year + 1, 1, 1) - timedelta(days=1)
        return date(year, month + 1, 1) - timedelta(days=1)

    @staticmethod
    def _month_range_from_offset(months_ago: int) -> tuple[date, date]:
        """Return first/last day of the month ``months_ago`` from today.

        Args:
            months_ago: Month offset where ``0`` means current month,
                ``1`` means previous month, etc.

        Returns:
            Tuple of ``(start_date, end_date)`` for the resolved month.
        """
        if months_ago <= 0:
            months_ago = 0

        today = date.today()
        year = today.year
        month = today.month - months_ago
        while month <= 0:
            month += 12
            year -= 1

        start = date(year, month, 1)
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        end = next_month - timedelta(days=1)
        return start, end

    @staticmethod
    def _slice_period(
        frame: pd.DataFrame,
        start: date,
        end: date,
        category: str,
    ) -> pd.DataFrame:
        """Filter transactions by inclusive date range and optional category.

        Args:
            frame: Input transaction frame with normalized columns.
            start: Inclusive range start.
            end: Inclusive range end.
            category: Optional category substring filter (case-insensitive).

        Returns:
            Filtered DataFrame copy scoped to the requested period/category.
        """
        scoped = frame[
            (frame["Transaction Date"].dt.date >= start)
            & (frame["Transaction Date"].dt.date <= end)
        ].copy()

        if category:
            min_ratio = get_category_fuzzy_min_ratio()
            scoped = scoped[
                scoped["Transaction Category"]
                .astype(str)
                .map(
                    lambda candidate: category_matches(
                        category,
                        candidate,
                        min_ratio=min_ratio,
                    )
                )
            ]

        return scoped

    @staticmethod
    def _aggregate_spend(
        scoped: pd.DataFrame,
        aggregation_method: str,
        weekly_average: bool,
    ) -> float:
        """Aggregate spend for a scoped frame, optionally by weekly average.

        Args:
            scoped: Period/category-scoped transactions.
            aggregation_method: ``sum``/``total`` or ``average``/``avg``/``mean``.
            weekly_average: If True, divide aggregate value by unique ISO week count.

        Returns:
            Aggregate numeric value for the selected mode.
        """
        method = (aggregation_method or "sum").strip().lower()
        if method in {"avg", "mean"}:
            method = "average"
        elif method in {"total"}:
            method = "sum"
        elif method not in {"sum", "average"}:
            method = "sum"

        if method == "average":
            base_value = float(scoped["Transaction Total"].mean())
        else:
            base_value = float(scoped["Transaction Total"].sum())

        if not weekly_average:
            return base_value

        week_count = int(scoped["Transaction Date"].dt.isocalendar().week.nunique())
        if week_count <= 0:
            return 0.0
        return base_value / week_count

    def ask(
        self,
        question: str,
        validated_rows: list[dict[str, Any]],
        chat_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Invoke the LangChain agent synchronously and return a structured response.

        Args:
            question: The natural-language question from the user.
            validated_rows: List of validated transaction dicts to analyse.
            chat_history: Optional list of prior turns, each a dict with ``role``
                and ``text`` keys, used to maintain conversational context.

        Returns:
            A dict with keys ``answer`` (str), ``rowsScanned`` (int),
            ``toolUsed`` (bool), and ``confidence`` (str).
        """
        output = self.ask_with_schema(
            AgentInput(
                question=question,
                validated_rows=validated_rows,
                chat_history=chat_history,
            )
        )
        return output.to_dict()

    def ask_with_schema(self, payload: AgentInput) -> AgentOutput:
        """
        Invoke the tool-calling pipeline using typed input/output payloads.

        Args:
            payload: Structured input containing question, rows, and history.

        Returns:
            Structured ``AgentOutput`` containing answer and run metadata.
        """
        # Refresh the per-request data backing the tool closures.
        self._validated_rows = payload.validated_rows
        messages = self.messages_with_history(
            payload.question,
            payload.chat_history,
        )

        # Pass 1: let the tool-calling agent reason and produce tool outputs.
        result = self._agent.invoke({"messages": messages})
        agent_output = result.get("messages", [])

        # Pass 2: synthesize the final user-facing response from tool outputs + context.
        answer = self._build_agent_answer(
            question=payload.question,
            chat_history=payload.chat_history,
            agent_output=agent_output,
        )

        # Preserve legacy response metadata contract for downstream consumers.
        tool_names: list[str] = []
        for msg in agent_output:
            msg_type = str(getattr(msg, "type", "")).lower()
            if "tool" not in msg_type:
                continue

            raw_name = (
                getattr(msg, "name", None)
                or getattr(msg, "tool_name", None)
                or getattr(msg, "tool", None)
            )
            if isinstance(raw_name, str) and raw_name.strip():
                tool_names.append(raw_name.strip())

        used_tool = len(tool_names) > 0
        tool_name_value = ", ".join(dict.fromkeys(tool_names)) if used_tool else "none"

        logger.info(
            "helper_agent user_input=%r llm_output=%r tool_used=%s tool_name=%s",
            payload.question,
            answer,
            used_tool,
            tool_name_value,
        )

        final_output = AgentOutput(
            answer=answer,
            rowsScanned=len(payload.validated_rows),
            toolUsed=used_tool,
        )

        return final_output

    def stream_answer(
        self,
        question: str,
        validated_rows: list[dict[str, Any]],
        chat_history: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        """
        Yield answer tokens from the agent stream with a safe synchronous fallback.

        Streams token-by-token via LangGraph's ``stream_mode="messages"`` and
        filters to only the ``model`` node to avoid emitting tool-call artifacts.
        If streaming fails or yields nothing, falls back to a full ``ask`` call.

        Args:
            question: The natural-language question from the user.
            validated_rows: List of validated transaction dicts to analyse.
            chat_history: Optional list of prior turns for conversational context.

        Yields:
            Incremental text tokens that together form the complete answer.
        """
        # Make the current transaction data available to the tool closure
        self._validated_rows = validated_rows
        yielded_any = False
        messages = self.messages_with_history(question, chat_history)

        try:
            stream_iter = self._agent.stream(
                {"messages": messages},
                stream_mode="messages",
            )

            for event in stream_iter:
                # LangGraph may emit (msg, metadata) tuples or bare message objects
                if isinstance(event, tuple):
                    msg, metadata = event
                else:
                    msg, metadata = event, {}

                # Skip events from nodes other than the LLM (e.g. tool executor)
                node = str((metadata or {}).get("langgraph_node", ""))
                if node and node != "model":
                    continue

                token = self._content_to_text(getattr(msg, "content", ""))
                if not token:
                    continue

                yielded_any = True
                yield token
        except Exception:
            # Streaming is best-effort; reset the flag so the fallback runs
            yielded_any = False

        if yielded_any:
            return

        # Fall back to a blocking ask() call and yield the full answer as one token
        fallback = self.ask(question, validated_rows, chat_history=chat_history).get(
            "answer", ""
        )
        if fallback:
            yield fallback

    @staticmethod
    def _extract_tools_output(agent_output: list[Any]) -> tuple[list[str], str]:
        """Extract synthesis-ready tool outputs and a fallback answer text.

        Uses only the most recent tool message. If that tool content is JSON and
        includes ``results``, only ``results`` is passed to synthesis.

        Args:
            agent_output: Messages returned by the tool-calling agent run.

        Returns:
            Tuple of ``(tool_outputs, fallback_text)`` where ``tool_outputs`` are
            synthesis-ready tool strings and ``fallback_text`` is the plain text
            from the last agent message (if any).
        """
        tool_outputs: list[str] = []
        for msg in reversed(agent_output):
            msg_type = str(getattr(msg, "type", "")).lower()
            if "tool" not in msg_type:
                continue

            msg_text = HelperAgent._content_to_text(getattr(msg, "content", ""))
            if not msg_text:
                continue

            # Prefer a compact tool payload for synthesis: only pass "results" when present.
            try:
                payload = json.loads(msg_text)
            except Exception:
                tool_outputs.append(msg_text)
                break

            if isinstance(payload, dict) and "results" in payload:
                tool_outputs.append(json.dumps(payload.get("results", [])))
            else:
                tool_outputs.append(msg_text)
            break

        fallback_text = ""
        if agent_output:
            fallback_text = HelperAgent._content_to_text(
                getattr(agent_output[-1], "content", "")
            ).strip()

        return tool_outputs, fallback_text

    def _build_agent_answer(
        self,
        question: str,
        chat_history: list[dict[str, Any]] | None,
        agent_output: list[Any],
    ) -> str:
        """
        Build the final user-facing answer from context and tool outputs.

        Tool calls are expected to return numeric/structured content. This method
        feeds the original question, recent chat context, and tool outputs into
        a synthesis pass to produce the final natural-language response.

        Args:
            question: Original user question.
            chat_history: Prior conversation turns.
            agent_output: Messages produced by the tool-calling run.

        Returns:
            Final synthesized answer string.
        """
        tool_outputs, fallback_text = self._extract_tools_output(agent_output)

        if not tool_outputs:
            if fallback_text:
                return fallback_text
            return ""

        return self._synthesize_answer(
            question=question,
            chat_history=chat_history,
            tool_outputs=tool_outputs,
            fallback_text=fallback_text,
        )

    def _synthesize_answer(
        self,
        question: str,
        chat_history: list[dict[str, Any]] | None,
        tool_outputs: list[str],
        fallback_text: str,
    ) -> str:
        """Run the final answer synthesis pass from tool outputs and context."""
        history_lines = self.history_lines(chat_history, limit=10)

        synthesis_messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": ARVEE_ANSWER_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Recent chat context:\n{chr(10).join(history_lines) if history_lines else '(none)'}\n\n"
                    f"Tool outputs (JSON/text):\n{chr(10).join(tool_outputs)}\n\n"
                    "Now write the final answer to the user."
                ),
            },
        ]

        try:
            synthesis_result = self._agent.invoke({"messages": synthesis_messages})
            synthesis_msgs = synthesis_result.get("messages", [])
            if synthesis_msgs:
                synthesized = self._content_to_text(
                    getattr(synthesis_msgs[-1], "content", "")
                ).strip()
                if synthesized:
                    return synthesized
        except Exception:
            pass

        return fallback_text

    @staticmethod
    def _build_comparison_chart(
        scoped_1: pd.DataFrame,
        scoped_2: pd.DataFrame,
        label_1: str,
        label_2: str,
        aggregation_method: str,
    ) -> dict[str, Any]:
        """Build chart payload for two category distributions."""
        method = normalize_aggregation_method(aggregation_method)
        agg = "mean" if method == "average" else "sum"

        grouped_1 = (
            scoped_1.groupby("Transaction Category", dropna=False)["Transaction Total"]
            .agg(agg)
            .sort_values(ascending=False)
        )
        grouped_2 = (
            scoped_2.groupby("Transaction Category", dropna=False)["Transaction Total"]
            .agg(agg)
            .sort_values(ascending=False)
        )

        ordered_categories: list[str] = []
        for cat in list(grouped_1.index) + list(grouped_2.index):
            cat_text = str(cat)
            if cat_text not in ordered_categories:
                ordered_categories.append(cat_text)

        values_1 = [
            round(float(grouped_1.get(cat, 0.0)), 2) for cat in ordered_categories
        ]
        values_2 = [
            round(float(grouped_2.get(cat, 0.0)), 2) for cat in ordered_categories
        ]

        # Build a per-category comparison table alongside the chart.
        table_rows = []
        for i, cat in enumerate(ordered_categories):
            v1 = values_1[i]
            v2 = values_2[i]
            delta = round(v1 - v2, 2)
            pct = round((delta / v2) * 100.0, 1) if v2 != 0 else None
            table_rows.append(
                {
                    "category": cat,
                    "period_1": v1,
                    "period_2": v2,
                    "delta": delta,
                    "percent_change": pct,
                }
            )

        return {
            "type": "grouped_bar",
            "title": f"Spending by category: {label_1} vs {label_2}",
            "x": ordered_categories,
            "currency": "USD",
            "series": [
                {"name": label_1, "values": values_1},
                {"name": label_2, "values": values_2},
            ],
            "table": {
                "columns": ["Category", label_1, label_2, "Delta ($)", "Change (%)"],
                "rows": table_rows,
            },
        }

    @staticmethod
    def _build_spending_breakdown_chart(
        scoped: pd.DataFrame,
        aggregation_method: str,
        chart_type: str,
        title_prefix: str,
    ) -> dict[str, Any]:
        """Build single-period category chart payload for bar or pie display."""
        method = normalize_aggregation_method(aggregation_method)
        agg = "mean" if method == "average" else "sum"

        grouped = (
            scoped.groupby("Transaction Category", dropna=False)["Transaction Total"]
            .agg(agg)
            .sort_values(ascending=False)
        )

        labels = [str(cat) for cat in grouped.index]
        values = [round(float(val), 2) for val in grouped.values]
        selected_chart_type = (
            "pie" if str(chart_type or "").strip().lower() == "pie" else "bar"
        )

        # Include the top 5 categories breakdown beneath the chart.
        top_5 = grouped.head(5)
        top_categories = [
            {"category": str(cat), "value": round(float(val), 2)}
            for cat, val in top_5.items()
        ]

        return {
            "type": selected_chart_type,
            "title": title_prefix,
            "currency": "USD",
            "labels": labels,
            "values": values,
        }

    @staticmethod
    def _to_frame(validated_rows: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Normalize validated-row records into a typed DataFrame ready for analysis.

        Ensures the four expected columns are always present, coerces numeric and
        date columns to their proper dtypes, and drops rows where either value
        could not be parsed (which would make aggregation unreliable).

        Args:
            validated_rows: List of transaction dicts as stored in the session.
                Each dict should contain Transaction Business Name,
                Transaction Total, Transaction Date, and
                Transaction Category keys.

        Returns:
            A clean DataFrame with the four expected columns, or an empty
            DataFrame with those columns if *validated_rows* is empty/invalid.
        """
        if not isinstance(validated_rows, list) or not validated_rows:
            return pd.DataFrame(
                [],
                columns=[
                    "Transaction Business Name",
                    "Transaction Total",
                    "Transaction Date",
                    "Transaction Category",
                ],
            )

        frame = pd.DataFrame(validated_rows).copy()

        # Guarantee all expected columns exist even if some rows omit them
        for col in [
            "Transaction Business Name",
            "Transaction Total",
            "Transaction Date",
            "Transaction Category",
        ]:
            if col not in frame.columns:
                frame[col] = None

        # Coerce to numeric; non-parseable values become NaN and are dropped below
        frame["Transaction Total"] = pd.to_numeric(
            frame["Transaction Total"], errors="coerce"
        )
        # Coerce to datetime; non-parseable values become NaT and are dropped below
        frame["Transaction Date"] = pd.to_datetime(
            frame["Transaction Date"], errors="coerce"
        )
        # Use 'Other' for missing categories so group-by operations stay consistent
        frame["Transaction Category"] = (
            frame["Transaction Category"].fillna("Other").astype(str)
        )

        # Drop rows where essential numeric/date values are unparseable
        frame = frame.dropna(subset=["Transaction Total", "Transaction Date"])
        return frame
