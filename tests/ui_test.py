"""Launch the UI with a pre-populated mock session for testing the HelperAgent."""

import random
import threading
import webbrowser
import os
from datetime import date, timedelta


TEST_SESSION_ID = "test-session-001"

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


def _mock_validated_transactions(count: int = 30) -> list[dict]:
    """Generate *count* realistic validated transaction rows."""
    random.seed(42)
    today = date.today()
    rows = []

    for i in range(count):
        biz, category = random.choice(BUSINESSES)
        total = round(random.uniform(3.50, 250.00), 2)
        tx_date = today - timedelta(days=random.randint(0, 45))

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


def seed_test_session(database) -> str:
    """Create (or overwrite) a test session in the database."""
    validated = _mock_validated_transactions(30)

    database.get_or_create_session(TEST_SESSION_ID)
    database.save_session_state(
        TEST_SESSION_ID,
        {
            "summary": "Mock session with 30 validated transactions for testing.",
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


if __name__ == "__main__":
    # Defer heavy web app imports until script execution time.
    from webui.app import app, database
    from flask import redirect, request

    session_id = seed_test_session(database)

    url = f"http://127.0.0.1:7860?testSession={session_id}"
    print(f"Open this URL to auto-load seeded data: {url}")

    @app.before_request
    def _redirect_root_to_seeded_session():
        if request.method != "GET":
            return None
        if request.path != "/":
            return None
        if request.args.get("testSession"):
            return None
        return redirect(f"/?testSession={session_id}")

    auto_open_browser = os.getenv("TEST_UI_OPEN_BROWSER", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if auto_open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    debug_enabled = os.getenv("TEST_UI_DEBUG", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    app.run(host="0.0.0.0", port=7860, debug=debug_enabled, use_reloader=False)
