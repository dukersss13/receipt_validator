import requests
import pandas as pd
from src.utils import load_exchange_rate_key
from datetime import datetime


access_key = load_exchange_rate_key()

class CurrencyConversionState(pd.DataFrame):
    amount: float
    currency: str
    date: str
    usd_value: float | None


def convert_currency_to_usd(entry: dict) -> float:
    """
    Converts a given amount in foreign currency to USD using the exchange rate on the specified date.
    """
    # Set API Endpoint and Parameters

    # Ensure date is in YYYY-MM-DD format
    date_str = entry["date"]
    try:
        # Try parsing with common formats, fallback to original if already correct
        parsed_date = datetime.strptime(date_str, "%m-%d-%Y")
    except ValueError:
        # Try alternative common formats
        for fmt in ("%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y"):
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unrecognized date format: {date_str}")
    formatted_date = parsed_date.strftime("%Y-%m-%d")

    params = {
        "access_key": access_key,
        "from": entry["currency"].upper(),
        "to": "USD",
        "amount": entry["total"],
        "date": formatted_date,
    }
    response = requests.get("https://api.exchangerate.host/convert", params=params)
    data = response.json()

    if not data.get("success", False):
        currency_val = -1
    else:
        currency_val = round(data["result"], 2)

    return currency_val
