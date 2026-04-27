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
    """
    Normalise a date string of various formats to ``YYYY-MM-DD`` for exchange rate API calls.

    Tries ``%m-%d-%Y`` first, then falls back through a list of other common formats.

    Args:
        date_str: Input date string in any of the supported formats.

    Returns:
        Date string formatted as ``"YYYY-MM-DD"``.

    Raises:
        ValueError: If the date string does not match any recognised format.
    """
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
    """
    Build the query-parameter dict for an exchangerate.host ``/convert`` API call.

    Args:
        entry: Transaction dict containing ``currency``, ``total``, and ``date`` keys.

    Returns:
        Dict of query parameters ready to pass to ``requests.get()``.
    """
    return {
        "access_key": access_key,
        "from": str(entry["currency"]).upper(),
        "to": "USD",
        "amount": entry["total"],
        "date": _normalize_date(str(entry["date"])),
    }


def convert_currency_to_usd(entry: dict) -> float:
    """
    Convert a transaction amount from a foreign currency to USD.

    Uses the historical exchange rate for the transaction date via the
    exchangerate.host API. USD entries are returned unchanged without an API call.

    Args:
        entry: Dict containing ``currency`` (ISO 4217 code), ``total`` (numeric
            amount), and ``date`` (date string in any supported format).

    Returns:
        The converted amount in USD rounded to 2 decimal places, or ``-1`` if
        the API call was unsuccessful.
    """
    currency = str(entry.get("currency", "USD")).upper()
    amount = float(entry.get("total", 0.0))

    # No conversion needed for USD entries.
    if currency == "USD":
        return round(amount, 2)

    params = _build_params(entry)
    response = requests.get(CONVERT_URL, params=params, timeout=20)
    data = response.json()

    # Return -1 as a sentinel for failed API responses to distinguish from zero
    if not data.get("success", False):
        currency_val = -1
    else:
        currency_val = round(data["result"], 2)

    return currency_val


async def convert_currency_to_usd_async(entry: dict) -> float:
    """
    Async wrapper for currency conversion using a non-blocking thread delegation.

    Args:
        entry: Transaction dict with ``currency``, ``total``, and ``date`` keys.

    Returns:
        The converted USD amount as returned by ``convert_currency_to_usd()``.
    """
    return await asyncio.to_thread(convert_currency_to_usd, entry)


async def convert_entries_to_usd_async(
    entries: list[dict], max_concurrency: int = 8
) -> list[float]:
    """
    Convert many transaction entries to USD concurrently with bounded fan-out.

    Uses a semaphore to cap the number of simultaneous exchange-rate API calls
    and avoids overwhelming the external service.

    Args:
        entries: List of transaction dicts each containing ``currency``, ``total``,
            and ``date`` keys.
        max_concurrency: Maximum number of concurrent conversion requests.
            Defaults to 8.

    Returns:
        List of converted USD amounts in the same order as *entries*.
    """
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def run_with_limit(entry: dict) -> float:
        async with semaphore:
            return await convert_currency_to_usd_async(entry)

    tasks = [run_with_limit(entry) for entry in entries]
    return await asyncio.gather(*tasks)
