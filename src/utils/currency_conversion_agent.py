import requests
import asyncio
import pandas as pd
from src.utils.utils import load_exchange_rate_key
from datetime import datetime


access_key = load_exchange_rate_key()
CONVERT_URL = "https://api.exchangerate.host/convert"

class CurrencyConversionState(pd.DataFrame):
    amount: float
    currency: str
    date: str
    usd_value: float | None


def _normalize_date(date_str: str) -> str:
    """Normalize input dates to YYYY-MM-DD for exchange rate API calls."""
    try:
        parsed_date = datetime.strptime(date_str, "%m-%d-%Y")
    except ValueError:
        for fmt in ("%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unrecognized date format: {date_str}")

    return parsed_date.strftime("%Y-%m-%d")


def _build_params(entry: dict) -> dict:
    return {
        "access_key": access_key,
        "from": str(entry["currency"]).upper(),
        "to": "USD",
        "amount": entry["total"],
        "date": _normalize_date(str(entry["date"])),
    }


def convert_currency_to_usd(entry: dict) -> float:
    """
    Converts a given amount in foreign currency to USD using the exchange rate on the specified date.
    """
    params = _build_params(entry)
    response = requests.get(CONVERT_URL, params=params, timeout=20)
    data = response.json()

    if not data.get("success", False):
        currency_val = -1
    else:
        currency_val = round(data["result"], 2)

    return currency_val


async def convert_currency_to_usd_async(entry: dict) -> float:
    """Async wrapper for currency conversion using non-blocking thread delegation."""
    return await asyncio.to_thread(convert_currency_to_usd, entry)


async def convert_entries_to_usd_async(
    entries: list[dict], max_concurrency: int = 8
) -> list[float]:
    """Convert many entries concurrently with bounded fan-out."""
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def run_with_limit(entry: dict) -> float:
        async with semaphore:
            return await convert_currency_to_usd_async(entry)

    tasks = [run_with_limit(entry) for entry in entries]
    return await asyncio.gather(*tasks)
