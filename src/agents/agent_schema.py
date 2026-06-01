from dataclasses import dataclass
from enum import Enum
from typing import Any


class Tools(Enum):
    """Supported helper-agent tool identifiers."""
    SPENDING_BREAKDOWN = "spending_breakdown"
    COMPARE_SPENDING_PERIODS = "compare_spending_periods"


@dataclass(slots=True)
class AgentInput:
    """Typed request payload for AgentTools invocations.

    Attributes:
        question: Raw user question to answer.
        validated_rows: Normalized validated transaction rows available for tools.
        chat_history: Optional prior turns in ``{"role", "text"}`` shape.
    """

    question: str
    validated_rows: list[dict[str, Any]]
    chat_history: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class AgentOutput:
    """Typed response payload returned by AgentTools.

    Attributes:
        answer: Final natural-language answer shown to the user.
        rowsScanned: Number of validated rows available for analysis.
        toolUsed: Whether at least one tool was called in the agent run.
        confidence: Coarse confidence label for UI consumption.
    """

    answer: str
    rowsScanned: int
    toolUsed: bool
    confidence: str = "high"

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass output into the legacy dict response shape.

        Returns:
            A JSON-serializable dict used by existing API/UI call sites.
        """
        return {
            "answer": self.answer,
            "rowsScanned": self.rowsScanned,
            "toolUsed": self.toolUsed,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class RouterInput:
    """Typed request payload for RouterAgent planning.

    Attributes:
        question: Raw user question to route.
        chat_history: Optional prior turns in ``{"role", "text"}`` shape.
    """

    question: str
    chat_history: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class RouterPlan:
    """Structured router decision returned before downstream dispatch.

    Attributes:
        tool_name: Target tool name for routed execution.
        tool_params: Extracted, normalized params passed to the chosen tool.
        needs_clarification: Whether the request is too ambiguous to execute.
        clarification_question: Concise question to ask when clarification is needed.
        confidence: Coarse confidence label for UI/telemetry usage.
    """
    tool_name: Tools | None
    tool_params: dict[str, Any]
    needs_clarification: bool = False
    clarification_question: str = ""
    confidence: str = "high"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable router plan dictionary."""
        return {
            "toolName": self.tool_name.value if self.tool_name else None,
            "toolParams": self.tool_params,
            "needsClarification": self.needs_clarification,
            "clarificationQuestion": self.clarification_question,
            "confidence": self.confidence,
        }
