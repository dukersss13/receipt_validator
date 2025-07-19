from langchain_openai import ChatOpenAI

from src.utils.utils import setup_openai
import pandas as pd


setup_openai()
model = ChatOpenAI(model="gpt-4o")

foreign_currency_path = "data/foreign_currency"


def detect_foreign_currency(data: pd.DataFrame):
    """
    Detects foreign currency in the provided DataFrame.
    """
    foreign_currencies = data[data["Currency"] != "USD"]

    return foreign_currencies


def test_currency_conversion_agent():
    # Test the currency conversion agent with a sample receipt text

    # Work on Currency Conversion Agent
    # Tools: 
    # 1. Build Agents Graph for Image to Text, Currency Detection, and Currency Conversion
    from graph import run_graph

    run_graph()
