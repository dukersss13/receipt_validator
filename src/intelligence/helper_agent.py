import re
import json
import logging
from datetime import date, timedelta
from typing import Any, Iterator

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import BaseTool, tool
from src.intelligence.agent_schema import AgentInput, AgentOutput
from src.intelligence.llm_base import LLMBase
from src.prompts.arvee_prompts import ARVEE_ANSWER_PROMPT, ARVEE_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


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
                return json.dumps({"status": "no_data"})

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
                return json.dumps(
                    {
                        "status": "no_results",
                        "category_filter": category or None,
                        "this_month": this_month,
                    }
                )

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
                return json.dumps(
                    {
                        "status": "ok",
                        "type": "top_categories",
                        "aggregation_method": method,
                        "this_month": this_month,
                        "category_filter": category or None,
                        "top_n": int(max(1, top_n)),
                        "results": [
                            {"category": str(cat), "value": round(float(val), 2)}
                            for cat, val in top.items()
                        ],
                    }
                )
            elif method == "average":
                return json.dumps(
                    {
                        "status": "ok",
                        "type": "average",
                        "this_month": this_month,
                        "category_filter": category or None,
                        "value": round(float(scoped["Transaction Total"].mean()), 2),
                    }
                )
            else:
                total = float(scoped["Transaction Total"].sum())
                return json.dumps(
                    {
                        "status": "ok",
                        "type": "total",
                        "this_month": this_month,
                        "category_filter": category or None,
                        "value": round(total, 2),
                    }
                )

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
                return json.dumps({"status": "no_data"})

            start_1, end_1, label_1 = HelperAgent._resolve_period(period_1)
            start_2, end_2, label_2 = HelperAgent._resolve_period(period_2)

            scoped_1 = HelperAgent._slice_period(frame, start_1, end_1, category)
            scoped_2 = HelperAgent._slice_period(frame, start_2, end_2, category)

            if scoped_1.empty or scoped_2.empty:
                return json.dumps(
                    {
                        "status": "insufficient_data",
                        "period_1": label_1,
                        "period_2": label_2,
                    }
                )

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
            percent_change = None
            if value_2 == 0:
                percent_change = None
            else:
                percent_change = round((delta / value_2) * 100.0, 1)

            metric_label = "weekly average" if weekly_average else "value"
            return json.dumps(
                {
                    "status": "ok",
                    "type": "comparison",
                    "metric": metric_label,
                    "category_filter": category or None,
                    "period_1": {
                        "token": period_1,
                        "label": label_1,
                        "value": round(value_1, 2),
                    },
                    "period_2": {
                        "token": period_2,
                        "label": label_2,
                        "value": round(value_2, 2),
                    },
                    "delta": round(delta, 2),
                    "percent_change": percent_change,
                }
            )

        return compare_spending_periods

    @staticmethod
    def _resolve_period(period_token: str) -> tuple[date, date, str]:
        """Resolve a user period token into a concrete date range.

        Args:
            period_token: Period expression (for example: ``this_month``,
                ``last_month``, ``2_months_ago``, ``YYYY-MM``).

        Returns:
            Tuple of ``(start_date, end_date, human_label)``.
        """
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
        messages = self._add_context_to_messages(payload.question, payload.chat_history)

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
        messages = self._add_context_to_messages(question, chat_history)

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
    def _add_context_to_messages(
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

    @staticmethod
    def _get_latest_chat_history(
        chat_history: list[dict[str, Any]] | None,
        limit: int = 10,
    ) -> list[str]:
        """Return the latest valid chat-history lines for synthesis context.

        Args:
            chat_history: Prior conversation turns in ``{"role", "text"}`` shape.
            limit: Maximum number of latest turns to include.

        Returns:
            List of ``"role: text"`` lines containing only valid user/assistant turns.
        """
        history_lines: list[str] = []
        if not isinstance(chat_history, list) or not chat_history:
            return history_lines

        for item in chat_history[-limit:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            text = str(item.get("text", "") or "").strip()
            if role in {"user", "assistant"} and text:
                history_lines.append(f"{role}: {text}")

        return history_lines

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

        history_lines = self._get_latest_chat_history(chat_history, limit=10)

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

        if fallback_text:
            return fallback_text
        return ""

    @staticmethod
    def _used_tool(agent_output: list[Any]) -> bool:
        """
        Return True if any tool-call message is present in the agent's message list.

        Args:
            agent_output: The ``agent_output`` list returned by ``agent.invoke()``.

        Returns:
            ``True`` if the agent invoked at least one tool; ``False`` otherwise.
        """
        for msg in agent_output:
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
