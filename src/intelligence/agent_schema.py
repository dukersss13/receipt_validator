from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AgentInput:
    """Typed request payload for HelperAgent invocations.

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
    """Typed response payload returned by HelperAgent.

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
