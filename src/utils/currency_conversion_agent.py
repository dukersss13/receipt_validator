import requests
from src.utils.utils import load_exchange_rate_key
from datetime import datetime

access_key = load_exchange_rate_key()
CONVERT_URL = "https://api.exchangerate.host/convert"


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
