"""Launch the UI with a pre-populated mock session for testing the AgentTools."""

import functools
import http.server
import os
import random
import socket
import socketserver
import threading
import webbrowser
from datetime import date, timedelta

import pandas as pd

TEST_SESSION_ID = "test-session-001"
DEFAULT_BACKEND_PORT = 7860
DEFAULT_FRONTEND_PORT = 8000
DEFAULT_WEBUI_DIR = os.path.join(os.path.dirname(__file__), "..", "arvee_web_ui")

BUSINESSES = [
    ("Starbucks", "Food & Drink"),
    ("Chipotle", "Food & Drink"),
    ("Uber", "Transport"),
    ("Lyft", "Transport"),
    ("Amazon", "Shopping"),
    ("Target", "Shopping"),
    ("Whole Foods", "Grocery"),
    ("Trader Joe's", "Grocery"),
    ("Netflix", "Entertainment"),
    ("Spotify", "Entertainment"),
    ("Shell Gas", "Gas"),
    ("Chevron", "Gas"),
    ("Delta Airlines", "Travel"),
    ("Hilton Hotels", "Travel"),
    ("CVS Pharmacy", "Health"),
    ("Planet Fitness", "Health"),
    ("Verizon", "Utilities"),
    ("Con Edison", "Utilities"),
    ("WeWork", "Office"),
    ("Staples", "Office"),
]


def _mock_validated_transactions(count: int = 500) -> list[dict]:
    """Generate *count* realistic validated transaction rows."""
    random.seed(42)
    today = date.today()
    rows = []

    for i in range(count):
        biz, category = random.choice(BUSINESSES)
        total = round(random.uniform(3.50, 250.00), 2)
        # Force coverage across a rolling 4-month window (about 120 days).
        tx_date = today - timedelta(days=(i % 120))

        rows.append(
            {
                "Transaction Business Name": biz,
                "Transaction Total": total,
                "Transaction Date": tx_date.isoformat(),
                "Transaction Category": category,
                "Proof Business Name": biz,
                "Proof Total": total,
                "Proof Date": tx_date.isoformat(),
                "Result": "Validated",
            }
        )

    return rows


def seed_test_session(database, user_id: str = "anonymous") -> str:
    """Create (or overwrite) a test session in the database."""
    validated = _mock_validated_transactions(500)
    transactions_df = pd.DataFrame(
        [
            {
                "business_name": r["Transaction Business Name"],
                "total": r["Transaction Total"],
                "date": r["Transaction Date"],
                "currency": "USD",
            }
            for r in validated
        ]
    )
    proofs_df = pd.DataFrame(
        [
            {
                "business_name": r["Proof Business Name"],
                "total": r["Proof Total"],
                "date": r["Proof Date"],
                "currency": "USD",
            }
            for r in validated
        ]
    )

    database.get_or_create_session(TEST_SESSION_ID, user_id=user_id)
    # Seed canonical inputs so /api/session/<id> can always load rows.
    database.save_session_inputs(
        TEST_SESSION_ID,
        transactions_df,
        proofs_df,
        user_id=user_id,
    )
    database.save_session_state(
        TEST_SESSION_ID,
        {
            "summary": "Mock session with 500 validated transactions over 4 months for testing.",
            "loadedTransactions": [
                {
                    "business_name": r["Transaction Business Name"],
                    "total": r["Transaction Total"],
                    "date": r["Transaction Date"],
                    "currency": "USD",
                    "category": r["Transaction Category"],
                }
                for r in validated
            ],
            "loadedProofs": [
                {
                    "business_name": r["Proof Business Name"],
                    "total": r["Proof Total"],
                    "date": r["Proof Date"],
                    "currency": "USD",
                    "category": r["Transaction Category"],
                }
                for r in validated
            ],
            "validatedTransactions": validated,
            "discrepancies": [],
            "unmatchedTransactions": [],
            "unmatchedProofs": [],
            "recommendations": [],
            "chatHistory": [],
        },
    )

    print(
        f"Seeded test session '{TEST_SESSION_ID}' with {len(validated)} validated transactions."
    )
    return TEST_SESSION_ID


def _resolve_port(default_port: int = DEFAULT_BACKEND_PORT) -> int:
    """Return default_port when available, else choose an open ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if probe.connect_ex(("127.0.0.1", default_port)) != 0:
            return default_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as fallback:
        fallback.bind(("127.0.0.1", 0))
        return int(fallback.getsockname()[1])


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def _start_static_server(directory: str, default_port: int = DEFAULT_FRONTEND_PORT) -> tuple[int, socketserver.TCPServer]:
    port = _resolve_port(default_port)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    server = _ThreadingServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Static web UI server listening on http://127.0.0.1:{port}")
    return port, server


if __name__ == "__main__":
    backend_port = _resolve_port(DEFAULT_BACKEND_PORT)
    frontend_port = _resolve_port(DEFAULT_FRONTEND_PORT)
    webui_dir = os.path.normpath(DEFAULT_WEBUI_DIR)
    static_port, _static_server = _start_static_server(webui_dir, frontend_port)

    os.environ["ARVEE_CORS_ORIGINS"] = (
        f"http://127.0.0.1:{static_port},http://localhost:{static_port}"
    )
    os.environ.setdefault("ARVEE_REQUIRE_USER_ID", "0")

    from backend_app import app, database
    from flask import redirect, request

    session_id = seed_test_session(database, user_id="anonymous")
    backend_base_url = f"http://127.0.0.1:{backend_port}"
    url = (
        f"http://127.0.0.1:{static_port}/index.html?testSession={session_id}"
        f"&apiBaseUrl={backend_base_url}"
    )
    host_friendly_url = (
        f"http://localhost:{static_port}/index.html?testSession={session_id}"
        f"&apiBaseUrl={backend_base_url}"
    )
    print(f"Open this URL to auto-load seeded data: {host_friendly_url}")
    print(f"Container-local URL: {url}")
    if backend_port != DEFAULT_BACKEND_PORT:
        print(f"Port {DEFAULT_BACKEND_PORT} is busy; using fallback port {backend_port}.")

    @app.before_request
    def _redirect_root_to_seeded_session():
        if request.method != "GET":
            return None
        if request.path != "/":
            return None
        if request.args.get("testSession"):
            return None
        return redirect(f"/?testSession={session_id}")

    debug_enabled = os.getenv("TEST_UI_DEBUG", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    auto_open_browser = os.getenv("TEST_UI_OPEN_BROWSER", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if auto_open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host="0.0.0.0", port=backend_port, debug=debug_enabled, use_reloader=False)
