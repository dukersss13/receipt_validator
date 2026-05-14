import re
from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.agents.agent_schema import AgentTool
from src.agents.agent_utils import (
    MONTH_NAME_TO_NUMBER,
    category_matches,
    get_category_fuzzy_min_ratio,
    normalize_aggregation_method,
    normalize_period_token,
)


class AgentTools:
    """
    Deterministic tool execution layer for ArVee analytics.
    """

    def __init__(self, validated_rows: list[dict[str, Any]] | None = None) -> None:
        self._validated_rows = validated_rows or []
        self._category_fuzzy_min_ratio = get_category_fuzzy_min_ratio()

    def set_validated_rows(self, validated_rows: list[dict[str, Any]]) -> None:
        """Replace the rows used by deterministic tool execution."""
        self._validated_rows = validated_rows

    def execute_tool(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a supported tool by name and return structured output."""
        selected_tool = AgentTool.from_value(tool_name)
        params = dict(tool_params or {})

        if selected_tool is AgentTool.SPENDING_BREAKDOWN:
            return self.execute_spending_breakdown(
                category=str(params.get("category", "") or ""),
                this_month=bool(params.get("this_month", False)),
                aggregation_method=str(
                    params.get("aggregation_method", "sum") or "sum"
                ),
                top_n=int(params.get("top_n", 0) or 0),
                period=params.get("period"),
                include_chart=bool(params.get("include_chart", False)),
                chart_type=str(params.get("chart_type", "bar") or "bar"),
            )

        return self.execute_compare_spending_periods(
            period_1=params.get("period_1", "this_month"),
            period_2=params.get("period_2", "last_month"),
            category=str(params.get("category", "") or ""),
            aggregation_method=str(params.get("aggregation_method", "sum") or "sum"),
            weekly_average=bool(params.get("weekly_average", False)),
            include_chart=bool(params.get("include_chart", False)),
        )

    @staticmethod
    def render_answer(tool_name: str, tool_output: dict[str, Any]) -> str:
        """Render deterministic answer text from a structured tool output."""
        selected_tool = AgentTool.from_value(tool_name)
        if selected_tool is AgentTool.COMPARE_SPENDING_PERIODS:
            return AgentTools._render_comparison_answer(tool_output)
        return AgentTools._render_spending_answer(tool_output)

    @staticmethod
    def _render_spending_answer(tool_output: dict[str, Any]) -> str:
        status = str(tool_output.get("status", "") or "").strip().lower()
        if status == "no_data":
            return "No validated transactions are available yet."
        if status == "no_results":
            suggestions = tool_output.get("timeframe_suggestions") or []
            if isinstance(suggestions, list) and suggestions:
                joined = "; ".join(str(item) for item in suggestions[:3])
                return (
                    "No transactions matched that timeframe. "
                    f"Try one of these queries: {joined}."
                )
            return "No transactions matched that filter for the selected period."

        has_chart = isinstance(tool_output.get("chart"), dict)
        result_type = str(tool_output.get("type", "") or "").strip().lower()

        if has_chart:
            return "Here is your spending chart and breakdown table."

        if result_type == "top_categories":
            results = tool_output.get("results") or []
            if isinstance(results, list) and results:
                first = results[0]
                category = str(first.get("category", "") or "").strip() or "Other"
                value = float(first.get("value", 0.0) or 0.0)
                return (
                    f"Top category is {category} at ${value:,.2f}. "
                    f"Returned {len(results)} ranked categories."
                )
            return "No ranked categories were found for that request."

        value = float(tool_output.get("value", 0.0) or 0.0)
        if result_type == "average":
            return f"Average spending is ${value:,.2f}."
        return f"Total spending is ${value:,.2f}."

    @staticmethod
    def _render_comparison_answer(tool_output: dict[str, Any]) -> str:
        status = str(tool_output.get("status", "") or "").strip().lower()
        if status == "no_data":
            return "No validated transactions are available yet."
        if status == "insufficient_data":
            period_1 = str(tool_output.get("period_1", "period 1") or "period 1")
            period_2 = str(tool_output.get("period_2", "period 2") or "period 2")
            return f"There is not enough data to compare {period_1} and {period_2}."

        period_1 = tool_output.get("period_1") or {}
        period_2 = tool_output.get("period_2") or {}
        label_1 = str(period_1.get("label", "period 1") or "period 1")
        label_2 = str(period_2.get("label", "period 2") or "period 2")
        value_1 = float(period_1.get("value", 0.0) or 0.0)
        value_2 = float(period_2.get("value", 0.0) or 0.0)
        delta = float(tool_output.get("delta", 0.0) or 0.0)
        pct = tool_output.get("percent_change")

        has_chart = isinstance(tool_output.get("chart"), dict)
        if has_chart:
            return f"Here is your period comparison chart for {label_1} vs {label_2}."

        if pct is None:
            return (
                f"{label_1}: ${value_1:,.2f}; {label_2}: ${value_2:,.2f}; "
                f"delta: ${delta:,.2f}."
            )

        return (
            f"{label_1}: ${value_1:,.2f}; {label_2}: ${value_2:,.2f}; "
            f"delta: ${delta:,.2f} ({float(pct):.1f}%)."
        )

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
        frame = AgentTools._to_frame(self._validated_rows)
        if frame.empty:
            return {"status": "no_data"}

        scoped = frame.copy()
        requested_label: str | None = None
        if period is not None:
            start, end, requested_label = AgentTools._resolve_period(
                period, frame=frame
            )
            scoped = AgentTools._slice_period(scoped, start, end, category="")
        elif this_month:
            now = pd.Timestamp(date.today())
            requested_label = "this month"
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
            timeframe_requested = bool(period is not None or this_month)
            return {
                "status": "no_results",
                "category_filter": category or None,
                "this_month": bool(this_month),
                "period": period,
                "requested_timeframe_label": requested_label,
                "timeframe_requested": timeframe_requested,
                "timeframe_suggestions": (
                    AgentTools._build_timeframe_suggestions(frame, category)
                    if timeframe_requested
                    else []
                ),
            }

        method = normalize_aggregation_method(aggregation_method)
        top_n_value = max(0, int(top_n or 0))

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
                payload["chart"] = AgentTools._build_spending_breakdown_chart(
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
                payload["chart"] = AgentTools._build_spending_breakdown_chart(
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
            payload["chart"] = AgentTools._build_spending_breakdown_chart(
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
        frame = AgentTools._to_frame(self._validated_rows)
        if frame.empty:
            return {"status": "no_data"}

        normalized_period_1 = normalize_period_token(period_1)
        normalized_period_2 = normalize_period_token(period_2)
        start_1, end_1, label_1 = AgentTools._resolve_period(
            normalized_period_1,
            frame=frame,
        )
        start_2, end_2, label_2 = AgentTools._resolve_period(
            normalized_period_2,
            frame=frame,
        )

        scoped_1 = AgentTools._slice_period(frame, start_1, end_1, category)
        scoped_2 = AgentTools._slice_period(frame, start_2, end_2, category)

        if scoped_1.empty or scoped_2.empty:
            return {
                "status": "insufficient_data",
                "period_1": label_1,
                "period_2": label_2,
            }

        method = normalize_aggregation_method(aggregation_method)
        weekly_average = bool(weekly_average)

        value_1 = AgentTools._aggregate_spend(
            scoped_1,
            aggregation_method=method,
            weekly_average=weekly_average,
        )
        value_2 = AgentTools._aggregate_spend(
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
                AgentTools._build_comparison_chart(
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

    @staticmethod
    def _build_timeframe_suggestions(
        frame: pd.DataFrame,
        category: str = "",
    ) -> list[str]:
        """Suggest valid timeframe queries based on available data."""
        if frame.empty:
            return []

        month_periods = frame["Transaction Date"].dropna().dt.to_period("M")
        unique_months = sorted(set(month_periods.tolist()))
        if not unique_months:
            return []

        latest = unique_months[-1]
        latest_label = latest.strftime("%Y-%m")
        category_prefix = f"for {category} " if category else ""

        suggestions = [
            f"total spending {category_prefix}in {latest_label}",
            f"compare spending {category_prefix}this month vs last month",
            f"total spending {category_prefix}across all transactions",
        ]

        if len(unique_months) > 1:
            prev_label = unique_months[-2].strftime("%Y-%m")
            suggestions[1] = (
                f"compare spending {category_prefix}in {latest_label} vs {prev_label}"
            )

        return suggestions

    @staticmethod
    def _resolve_period(
        period_token: Any,
        frame: pd.DataFrame | None = None,
    ) -> tuple[date, date, str]:
        """Resolve a user period token into a concrete date range."""
        today = date.today()

        if isinstance(period_token, dict):
            return AgentTools._resolve_period_spec(period_token, frame=frame)

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
            start = AgentTools._first_day_of_month(
                *AgentTools._shift_month(today.year, today.month, -(count - 1))
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
            start, end = AgentTools._month_range_from_offset(offset)
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
            end = AgentTools._last_day_of_month(year, month)
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
            start = AgentTools._first_day_of_month(
                *AgentTools._shift_month(today.year, today.month, -(n - 1))
            )
            return start, today, f"past {n} months"

        if kind in {"offset", "offset_window"}:
            offset = max(
                0, int(period_spec.get("offset", period_spec.get("months_ago", 0)) or 0)
            )
            span = max(1, int(period_spec.get("span", period_spec.get("n", 1)) or 1))
            if unit == "month":
                start_month = AgentTools._shift_month(
                    today.year, today.month, -(offset + span - 1)
                )
                end_month = AgentTools._shift_month(today.year, today.month, -offset)
                start = AgentTools._first_day_of_month(*start_month)
                end = AgentTools._last_day_of_month(*end_month)
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
                end = AgentTools._last_day_of_month(year, month)
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
                start = AgentTools._first_day_of_month(*selected[0])
                end = AgentTools._last_day_of_month(*selected[-1])
                label = f"{position} {len(selected)} months"
                return start, end, label

        token = str(period_spec.get("token", "this_month") or "this_month")
        return AgentTools._resolve_period(token, frame=frame)

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
        """Return first/last day of the month ``months_ago`` from today."""
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
        """Filter transactions by inclusive date range and optional category."""
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
        """Aggregate spend for a scoped frame, optionally by weekly average."""
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

        return {
            "type": selected_chart_type,
            "title": title_prefix,
            "currency": "USD",
            "labels": labels,
            "values": values,
        }

    @staticmethod
    def _to_frame(validated_rows: list[dict[str, Any]]) -> pd.DataFrame:
        """Normalize validated-row records into a typed DataFrame."""
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

        for col in [
            "Transaction Business Name",
            "Transaction Total",
            "Transaction Date",
            "Transaction Category",
        ]:
            if col not in frame.columns:
                frame[col] = None

        frame["Transaction Total"] = pd.to_numeric(
            frame["Transaction Total"], errors="coerce"
        )
        frame["Transaction Date"] = pd.to_datetime(
            frame["Transaction Date"], errors="coerce"
        )
        frame["Transaction Category"] = (
            frame["Transaction Category"].fillna("Other").astype(str)
        )

        return frame.dropna(subset=["Transaction Total", "Transaction Date"])
