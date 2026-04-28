import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterator

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import BaseTool, tool
from src.intelligence.llm_base import LLMBase


@dataclass(slots=True)
class AgentInput:
    """Schema for HelperAgent request payloads."""

    question: str
    validated_rows: list[dict[str, Any]]
    chat_history: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class AgentOutput:
    """Schema for HelperAgent response payloads."""

    answer: str
    rowsScanned: int
    toolUsed: bool
    confidence: str = "high"

    def to_dict(self) -> dict[str, Any]:
        """Return the output in the existing API response shape."""
        return {
            "answer": self.answer,
            "rowsScanned": self.rowsScanned,
            "toolUsed": self.toolUsed,
            "confidence": self.confidence,
        }


class HelperAgent(LLMBase):
    """
    Answer natural-language questions over validated transaction rows.

    Wraps a LangChain tool-calling agent backed by Gemini. Exposes both a
    synchronous ``ask`` method and a streaming ``stream_answer`` generator.
    """

    _SYSTEM_PROMPT = """
        You are ArVee, a concise personal finance assistant focused on validated transactions only.
        Answer spending questions using validated transactions and computed values only.
        If the user is unclear, ask for clarification.
        If the user does not specify a time frame, default to this month.
        If results are multiple, present results as a numbered list.
        If data is missing or a filter returns no rows, say so clearly and suggest a nearby alternative question.
        Do not invent transactions, dates, categories, or amounts.
        Answer directly to the user in second person. Never use first-person phrasing (for example: I, me, my, mine, we, our, us).
        Keep responses natural, friendly, short, direct, and numeric when possible.
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
        # Will be replaced on each ask/stream_answer call with fresh row data
        self._validated_rows: list[dict[str, Any]] = []
        self._agent = create_agent(
            model=self._model,
            tools=[
                self._breakdown_spending_tool(),
                self._compare_spending_periods_tool(),
            ],
            system_prompt=self._SYSTEM_PROMPT,
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
        # Capture self so the inner function can access live row data without being a method
        agent_ref = self

        @tool
        def spending_breakdown(
            category: str = "",
            this_month: bool = True,
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
            - if no timeframe is specified, keep ``this_month`` as True
            - leave ``category`` empty to include all categories
            - use ``aggregation_method`` as ``sum`` unless average is requested
            - set ``top_n`` only when a ranked top-N result is requested

            Args:
                category: Optional spending category to filter by (e.g. 'food', 'travel', 'grocery').
                          Leave empty to include all categories.
                this_month: If True, only include transactions from the current month.
                aggregation_method: Aggregation to apply: 'sum' (total) or 'average'.
                top_n: If greater than 0, return the top N categories by the selected aggregation.

            Returns:
                A text summary of the spending analysis.
            """
            frame = HelperAgent._to_frame(agent_ref._validated_rows)
            if frame.empty:
                return "No validated transactions are available yet."

            # Work on a copy so the original frame is never mutated between calls
            scoped = frame.copy()

            # Restrict to the current calendar month when requested
            if this_month:
                now = pd.Timestamp(date.today())
                scoped = scoped[
                    (scoped["Transaction Date"].dt.year == now.year)
                    & (scoped["Transaction Date"].dt.month == now.month)
                ]

            # Case-insensitive substring match on the category column
            if category:
                token = category.lower()
                scoped = scoped[
                    scoped["Transaction Category"]
                    .str.lower()
                    .str.contains(re.escape(token), na=False)
                ]

            if scoped.empty:
                return "No transactions found matching those filters."

            # Normalise synonyms so downstream logic only sees 'sum' or 'average'
            method = (aggregation_method or "sum").strip().lower()
            if method in {"avg", "mean"}:
                method = "average"
            elif method in {"total"}:
                method = "sum"
            elif method not in {"sum", "average"}:
                method = "sum"

            if top_n > 0:
                # Group by category, aggregate, then sort descending to get the top N
                grouped = (
                    scoped.groupby("Transaction Category", dropna=False)[
                        "Transaction Total"
                    ]
                    .agg("mean" if method == "average" else "sum")
                    .sort_values(ascending=False)
                )
                top = grouped.head(max(1, top_n))
                lines = [f"  {cat}: ${val:,.2f}" for cat, val in top.items()]
                label = "average spend" if method == "average" else "spend"
                return f"Top categories by {label}:\n" + "\n".join(lines)
            elif method == "average":
                return (
                    f"Average spend: ${float(scoped['Transaction Total'].mean()):,.2f}"
                )
            else:
                total = float(scoped["Transaction Total"].sum())
                # Build a human-readable scope suffix, e.g. "on food this month"
                scope_parts = []
                if category:
                    scope_parts.append(f"on {category}")
                if this_month:
                    scope_parts.append("this month")
                suffix = (" " + " ".join(scope_parts)) if scope_parts else ""
                return f"Total spend{suffix}: ${total:,.2f}"

        return spending_breakdown

    def _compare_spending_periods_tool(self) -> BaseTool:
        """
        Build and return the ``compare_spending_periods`` tool for the agent.

        Returns:
            A LangChain tool that compares two user-requested periods.
        """
        agent_ref = self

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
            frame = HelperAgent._to_frame(agent_ref._validated_rows)
            if frame.empty:
                return "No validated transactions are available yet."

            start_1, end_1, label_1 = HelperAgent._resolve_period(period_1)
            start_2, end_2, label_2 = HelperAgent._resolve_period(period_2)

            scoped_1 = HelperAgent._slice_period(frame, start_1, end_1, category)
            scoped_2 = HelperAgent._slice_period(frame, start_2, end_2, category)

            if scoped_1.empty or scoped_2.empty:
                return "Insufficient data for one or both requested periods."

            value_1 = HelperAgent._aggregate_spend(
                scoped_1,
                aggregation_method=aggregation_method,
                weekly_average=weekly_average,
            )
            value_2 = HelperAgent._aggregate_spend(
                scoped_2,
                aggregation_method=aggregation_method,
                weekly_average=weekly_average,
            )

            delta = value_1 - value_2
            if value_2 == 0:
                percent_part = "change: n/a (baseline is $0.00)"
            else:
                percent = (delta / value_2) * 100.0
                percent_part = f"change: {percent:+.1f}%"

            metric_label = "weekly average" if weekly_average else "value"
            prefix = f" for {category}" if category else ""
            return (
                f"Comparison{prefix} ({metric_label}):\n"
                f"- {label_1}: ${value_1:,.2f}\n"
                f"- {label_2}: ${value_2:,.2f}\n"
                f"- absolute change: ${delta:,.2f} ({label_1} - {label_2})\n"
                f"- {percent_part}"
            )

        return compare_spending_periods

    @staticmethod
    def _resolve_period(period_token: str) -> tuple[date, date, str]:
        """Resolve a period token into a [start, end] date range and label."""
        today = date.today()
        token = str(period_token or "this_month").strip().lower()

        if token == "this_month":
            start = date(today.year, today.month, 1)
            return start, today, "this month"

        if token == "last_month":
            first_of_this_month = date(today.year, today.month, 1)
            end = first_of_this_month - timedelta(days=1)
            start = date(end.year, end.month, 1)
            return start, end, "last month"

        months_ago_match = re.match(r"^(\d+)_months?_ago$", token)
        if months_ago_match:
            offset = int(months_ago_match.group(1))
            start, end = HelperAgent._month_range_from_offset(offset)
            return start, end, f"{offset} months ago"

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
    def _month_range_from_offset(months_ago: int) -> tuple[date, date]:
        """Return the first and last date of the month *months_ago* from today."""
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
        """Filter transactions by date range and optional category."""
        scoped = frame[
            (frame["Transaction Date"].dt.date >= start)
            & (frame["Transaction Date"].dt.date <= end)
        ].copy()

        if category:
            token = category.lower()
            scoped = scoped[
                scoped["Transaction Category"]
                .str.lower()
                .str.contains(re.escape(token), na=False)
            ]

        return scoped

    @staticmethod
    def _aggregate_spend(
        scoped: pd.DataFrame,
        aggregation_method: str,
        weekly_average: bool,
    ) -> float:
        """Aggregate spend for a period, optionally normalized by week count."""
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
        """Invoke the agent using ``AgentInput`` and return ``AgentOutput``."""
        # Make the current transaction data available to the tool closure
        self._validated_rows = payload.validated_rows
        messages = self._build_messages(payload.question, payload.chat_history)
        result = self._agent.invoke({"messages": messages})
        messages = result.get("messages", [])
        answer = self._extract_answer(messages)
        return AgentOutput(
            answer=answer,
            rowsScanned=len(payload.validated_rows),
            toolUsed=self._used_tool(messages),
        )

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
        messages = self._build_messages(question, chat_history)

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
    def _build_messages(
        question: str,
        chat_history: list[dict[str, Any]] | None,
        max_history_messages: int = 20,
    ) -> list[dict[str, str]]:
        """
        Assemble the message list for the agent, prepending up to *max_history_messages* prior turns.

        Args:
            question: The current user question to append as the final message.
            chat_history: Optional prior conversation turns. Each entry must be a
                dict with ``role`` (``"user"`` or ``"assistant"``) and ``text`` keys.
                Invalid or unknown roles are silently skipped.
            max_history_messages: Maximum number of history messages to include.
                Older messages beyond this limit are dropped.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts ready for the agent.
        """
        messages: list[dict[str, str]] = []
        if isinstance(chat_history, list) and chat_history:
            # Take only the most recent N turns to stay within context limits
            trimmed = chat_history[-max_history_messages:]
            for entry in trimmed:
                if not isinstance(entry, dict):
                    continue

                role = str(entry.get("role", "")).strip().lower()
                # Only include recognised roles to avoid confusing the model
                if role not in {"user", "assistant"}:
                    continue

                text = str(entry.get("text", "") or "").strip()
                if not text:
                    continue

                messages.append({"role": role, "content": text})

        # Always append the current question as the last user message
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """
        Extract a plain-text string from a LangChain message content value.

        LangChain message content can arrive in several shapes depending on the
        model and stream mode: a plain string, a list of string/dict chunks
        (multi-part content), or a single dict with a ``"text"`` key.

        Args:
            content: The raw ``content`` attribute of a LangChain message object.

        Returns:
            Concatenated text extracted from the content, or an empty string if
            no text could be found.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            # Multi-part content: concatenate all string chunks and dict text fields
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "".join(chunks)

        if isinstance(content, dict):
            text = content.get("text")
            return text if isinstance(text, str) else ""

        return ""

    @classmethod
    def _extract_answer(cls, messages: list[Any]) -> str:
        """
        Return the last AI message text from a completed agent run.

        Scans the message list in reverse so the most recent message wins.
        Falls back first to the last tool output, then to any non-empty message,
        and finally to a generic error string.

        Args:
            messages: The ``messages`` list returned by ``agent.invoke()``.

        Returns:
            The best available answer string extracted from the message list.
        """
        fallback_tool_text = ""
        fallback_any_text = ""

        for msg in reversed(messages):
            msg_type = str(getattr(msg, "type", "")).lower()
            msg_text = cls._content_to_text(getattr(msg, "content", ""))

            if not msg_text:
                continue

            # Record the first non-empty message encountered as a last-resort fallback
            if not fallback_any_text:
                fallback_any_text = msg_text

            # Prefer the most recent AI response message
            if "ai" in msg_type:
                return msg_text

            # Keep the most recent tool output as a secondary fallback
            if "tool" in msg_type and not fallback_tool_text:
                fallback_tool_text = msg_text

        if fallback_tool_text:
            return fallback_tool_text

        if fallback_any_text:
            return fallback_any_text

        return "I could not generate an answer."

    @staticmethod
    def _used_tool(messages: list[Any]) -> bool:
        """
        Return True if any tool-call message is present in the agent's message list.

        Args:
            messages: The ``messages`` list returned by ``agent.invoke()``.

        Returns:
            ``True`` if the agent invoked at least one tool; ``False`` otherwise.
        """
        for msg in messages:
            if "tool" in str(getattr(msg, "type", "")).lower():
                return True
        return False

    @staticmethod
    def _to_frame(validated_rows: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Normalize validated-row records into a typed DataFrame ready for analysis.

        Ensures the four expected columns are always present, coerces numeric and
        date columns to their proper dtypes, and drops rows where either value
        could not be parsed (which would make aggregation unreliable).

        Args:
            validated_rows: List of transaction dicts as stored in the session.
                Each dict should contain ``Transaction Business Name``,
                ``Transaction Total``, ``Transaction Date``, and
                ``Transaction Category`` keys.

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
