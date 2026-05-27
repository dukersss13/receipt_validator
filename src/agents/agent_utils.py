import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from pyhocon import ConfigFactory

MONTH_NAME_TO_NUMBER: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def normalize_aggregation_method(aggregation_method: str) -> str:
    """Normalize aggregation aliases to ``sum`` or ``average``."""
    method = (aggregation_method or "sum").strip().lower()
    if method in {"avg", "mean", "average"}:
        method = "average"
    elif method in {"total", "sum"}:
        method = "sum"

    return method


def normalize_period_token(period_token: Any) -> Any:
    """Normalize and lightly validate period tokens used by compare flows."""
    if isinstance(period_token, dict):
        return period_token

    token = str(period_token or "this_month").strip().lower()
    if token in {"this_month", "last_month"}:
        return token

    if token in {"now", "current", "current_month", "this month"}:
        return "this_month"

    if re.match(r"^past_\d+_months?$", token):
        return token

    if re.match(r"^\d+_months?_ago$", token):
        return token

    iso_month_match = re.match(r"^(\d{4})-(\d{2})$", token)
    if iso_month_match:
        month = int(iso_month_match.group(2))
        if 1 <= month <= 12:
            return token

    month_name_match = re.match(
        r"^(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)(?:\s+\d{4})?$",
        token,
    )
    if month_name_match:
        return token

    return "this_month"


def extract_first_json_object(raw_text: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from free-form model output text."""
    if not raw_text:
        return None

    candidates = [raw_text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
    candidates.extend(fenced)

    brace_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value

    return None


@lru_cache(maxsize=8)
def _load_app_config_cached(config_path: str) -> Any:
    """Parse and cache app config for lightweight shared settings access."""
    return ConfigFactory.parse_file(config_path)


def get_category_fuzzy_min_ratio(
    config_path: str = "config/config.conf",
    default: float = 0.72,
) -> float:
    """Return fuzzy category matching threshold from config, clamped to 0..1."""
    try:
        cfg = _load_app_config_cached(config_path)
        raw = cfg.get("matching.category_fuzzy_min_ratio", default)
        value = float(raw)
    except Exception:
        value = float(default)
    return max(0.0, min(1.0, value))


def _normalize_label(text: Any) -> str:
    """Normalize free-form labels for tolerant text matching."""
    value = str(text or "").strip().lower()
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def category_matches(query: str, candidate: str, min_ratio: float = 0.72) -> bool:
    """Return True when query and candidate should be treated as same category."""
    q = _normalize_label(query)
    c = _normalize_label(candidate)
    if not q or not c:
        return False

    if q == c:
        return True

    # Bi-directional containment handles cases like transport vs transportation.
    if q in c or c in q:
        return True

    q_compact = q.replace(" ", "")
    c_compact = c.replace(" ", "")
    if q_compact and c_compact and (q_compact in c_compact or c_compact in q_compact):
        return True

    q_tokens = [tok for tok in q.split(" ") if tok]
    c_tokens = [tok for tok in c.split(" ") if tok]
    if q_tokens and c_tokens:
        for q_tok in q_tokens:
            for c_tok in c_tokens:
                if q_tok.startswith(c_tok) or c_tok.startswith(q_tok):
                    return True

    ratio = SequenceMatcher(None, q_compact or q, c_compact or c).ratio()
    return ratio >= min_ratio
