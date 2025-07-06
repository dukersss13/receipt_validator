from typing import Optional, TypedDict
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.utils import setup_openai, load_exchange_rate_key
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
    from src.graph import run_graph

    run_graph()
