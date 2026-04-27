import pandas as pd
import os
import pytest

foreign_currency_path = "data/foreign_currency"


def detect_foreign_currency(data: pd.DataFrame):
    """
    Detects foreign currency in the provided DataFrame.
    """
    foreign_currencies = data[data["Currency"] != "USD"]

    return foreign_currencies


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ,
    reason="Skipping currency conversion test on GitHub Actions",
)
def test_currency_conversion_agent():
    # Test the currency conversion agent with a sample receipt text

    # Work on Currency Conversion Agent
    # Tools:
    # 1. Build Agents Graph for Image to Text, Currency Detection, and Currency Conversion
    from src.graph import run_graph

    run_graph()
