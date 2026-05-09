import os
import pytest

foreign_currency_path = "data/foreign_currency"


@pytest.mark.requires_llm
def test_currency_conversion_agent():
    # Test the currency conversion agent with a sample receipt text

    # Work on Currency Conversion Agent
    # Tools:
    # 1. Build Agents Graph for Image to Text, Currency Detection, and Currency Conversion
    from legacy.graph import run_graph

    run_graph()
