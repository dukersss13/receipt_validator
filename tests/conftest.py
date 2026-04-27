from datetime import date, timedelta
import random

import pandas as pd
import pytest


BUSINESSES = [
    ("Starbucks", "Food"),
    ("Chipotle", "Food"),
    ("Uber", "Transport"),
    ("Lyft", "Transport"),
    ("Amazon", "Shopping"),
    ("Target", "Shopping"),
    ("Whole Foods", "Food"),
    ("Trader Joe's", "Food"),
    ("Netflix", "Entertainment"),
    ("Spotify", "Entertainment"),
    ("Shell Gas", "Transport"),
    ("Chevron", "Transport"),
    ("Delta Airlines", "Travel"),
    ("Hilton Hotels", "Travel"),
    ("CVS Pharmacy", "Medical"),
    ("Planet Fitness", "Health & Fitness"),
    ("Equinox", "Health & Fitness"),
    ("OrangeTheory", "Health & Fitness"),
    ("Verizon", "Utilities"),
    ("Con Edison", "Utilities"),
    ("WeWork", "Other"),
    ("Staples", "Shopping"),
]


def generate_mock_validated_transactions(count: int = 30) -> list[dict]:
    """Generate deterministic validated rows used across unit tests."""
    random.seed(42)
    today = date.today()
    rows = []

    for _ in range(count):
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

    # Ensure the shared fixture always includes Health & Fitness samples.
    required_hf = ["Planet Fitness", "Equinox", "OrangeTheory"]
    for i, biz in enumerate(required_hf):
        if i >= len(rows):
            break
        rows[i]["Transaction Business Name"] = biz
        rows[i]["Proof Business Name"] = biz
        rows[i]["Transaction Category"] = "Health & Fitness"

    return rows


@pytest.fixture
def sample_validated_transactions_df() -> pd.DataFrame:
    return pd.DataFrame(generate_mock_validated_transactions(30))


@pytest.fixture
def sample_transactions_df(
    sample_validated_transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "business_name": sample_validated_transactions_df[
                "Transaction Business Name"
            ],
            "total": sample_validated_transactions_df["Transaction Total"],
            "date": sample_validated_transactions_df["Transaction Date"],
            "currency": "USD",
        }
    )


@pytest.fixture
def sample_proofs_df(sample_validated_transactions_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "business_name": sample_validated_transactions_df["Proof Business Name"],
            "total": sample_validated_transactions_df["Proof Total"],
            "date": sample_validated_transactions_df["Proof Date"],
            "currency": "USD",
        }
    )
