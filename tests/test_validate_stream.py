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


def test_validate_stream_emits_progress_and_done(monkeypatch: Any) -> None:
    def stub_pipeline(
        session_id: str,
        transactions: list[Any],
        proofs: list[Any],
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        if progress_callback is not None:
            progress_callback("Preparing uploaded files...", 5)
            progress_callback("Validation complete.", 100)
        return {
            "sessionId": session_id,
            "summary": "ok",
            "validatedTransactions": [],
            "discrepancies": [],
            "unmatchedTransactions": [],
            "unmatchedProofs": [],
            "recommendations": [],
        }

    monkeypatch.setattr(webapp_module, "_run_validation_pipeline", stub_pipeline)

    client = webapp_module.app.test_client()
    response = client.post(
        "/api/validate/stream",
        data={"sessionId": "session-1"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.get_data(as_text=True))
    assert [event["event"] for event in events] == [
        "start",
        "progress",
        "progress",
        "done",
    ]
    assert "Preparing uploaded files..." in events[1]["data"]
    assert "Validation complete." in events[2]["data"]
    assert '"sessionId": "session-1"' in events[3]["data"]


def test_validate_stream_emits_error_event(monkeypatch: Any) -> None:
    def stub_pipeline(
        session_id: str,
        transactions: list[Any],
        proofs: list[Any],
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        raise ValueError("Provide both transactions and proofs")

    monkeypatch.setattr(webapp_module, "_run_validation_pipeline", stub_pipeline)

    client = webapp_module.app.test_client()
    response = client.post(
        "/api/validate/stream",
        data={"sessionId": "session-2"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.get_data(as_text=True))
    assert [event["event"] for event in events] == ["start", "error"]
    assert "Provide both transactions and proofs" in events[1]["data"]
