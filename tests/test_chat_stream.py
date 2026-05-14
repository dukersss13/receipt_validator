import json
from typing import Any

from webui import app as webapp_module


def _parse_sse_events(payload: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in payload.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_name = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                data_lines.append(line[len("data: ") :].strip())
        events.append({"event": event_name, "data": "\n".join(data_lines)})
    return events


def test_chat_stream_emits_progress_when_validation_required(monkeypatch: Any) -> None:
    class StubDB:
        def load_session_state(self, session_id: str) -> dict[str, Any]:
            return {}

        def save_session_state(self, session_id: str, state: dict[str, Any]) -> None:
            return None

    monkeypatch.setattr(webapp_module, "database", StubDB())

    client = webapp_module.app.test_client()
    response = client.post(
        "/api/chat/ask/stream",
        json={"sessionId": "session-1", "message": "hello"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.get_data(as_text=True))

    assert [event["event"] for event in events] == [
        "start",
        "progress",
        "token",
        "done",
    ]
    assert "Validation is needed before chat can answer this." in events[1]["data"]
    assert (
        "Please upload your transactions and proofs in the Upload tab, then validate before asking questions"
        in events[2]["data"]
    )
    assert "How do I upload Transactions/Proofs?" in events[3]["data"]
    assert "How do I upload proofs?" not in events[3]["data"]


def test_chat_stream_validation_followup_after_upload(monkeypatch: Any) -> None:
    class StubDB:
        def load_session_state(self, session_id: str) -> dict[str, Any]:
            return {}

        def save_session_state(self, session_id: str, state: dict[str, Any]) -> None:
            return None

    monkeypatch.setattr(webapp_module, "database", StubDB())

    client = webapp_module.app.test_client()
    response = client.post(
        "/api/chat/ask/stream",
        json={"sessionId": "session-1", "message": "What should I do after upload?"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.get_data(as_text=True))
    assert "Run validation and wait for the results" in events[2]["data"]


def test_chat_stream_emits_progress_for_routed_answer(monkeypatch: Any) -> None:
    class StubDB:
        def __init__(self) -> None:
            self.saved_state: dict[str, Any] | None = None

        def load_session_state(self, session_id: str) -> dict[str, Any]:
            return {
                "validatedTransactions": [
                    {
                        "Transaction Business Name": "Store A",
                        "Transaction Total": 10.0,
                        "Transaction Date": "2026-05-01",
                        "Transaction Category": "Food",
                    }
                ],
                "chatHistory": [],
            }

        def save_session_state(self, session_id: str, state: dict[str, Any]) -> None:
            self.saved_state = state

    class StubRouter:
        def ask(
            self,
            message: str,
            validated_rows: list[dict[str, Any]],
            chat_history: list[dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            return {
                "answer": "Here is your latest spend summary.",
                "confidence": "high",
                "toolUsed": True,
                "toolName": "spending_breakdown",
                "needsClarification": False,
                "quickReplies": ["Show top categories"],
                "chart": None,
                "top_categories": [],
            }

    stub_db = StubDB()
    monkeypatch.setattr(webapp_module, "database", stub_db)
    monkeypatch.setattr(webapp_module, "RouterAgent", lambda: StubRouter())

    client = webapp_module.app.test_client()
    response = client.post(
        "/api/chat/ask/stream",
        json={"sessionId": "session-2", "message": "how much did I spend?"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.get_data(as_text=True))
    event_names = [event["event"] for event in events]

    assert event_names[0:4] == ["start", "progress", "progress", "progress"]
    assert event_names[-2:] == ["progress", "done"]
    token_events = [event for event in events if event["event"] == "token"]
    assert len(token_events) >= 2

    streamed_answer = "".join(
        json.loads(event["data"]).get("token", "") for event in token_events
    )
    assert streamed_answer.strip() == "Here is your latest spend summary."
    assert "Looking into your request..." in events[1]["data"]
    assert "Analyzing your validated transactions..." in events[2]["data"]
    assert "Finalizing the response..." in events[3]["data"]
    assert "Done." in events[-2]["data"]
    assert stub_db.saved_state is not None
