from src.data.data_reader import DataReader


def test_parse_extracted_payload_python_tuples_and_code_fences():
    raw = """```python
[("Coffee Shop", "12.34", "03/08/2026", "USD")]
```"""

    parsed = DataReader.parse_extracted_payload(raw)

    assert parsed == [("Coffee Shop", "12.34", "03/08/2026", "USD")]


def test_parse_extracted_payload_json_dict_and_default_currency():
    raw = """[
  {"business_name": "Cafe Rio", "total": "19.95", "date": "2026-03-08", "currency": "usd"},
  ["Boba House", "8.10", "03-08-2026"]
]"""

    parsed = DataReader.parse_extracted_payload(raw)

    assert parsed[0] == ("Cafe Rio", "19.95", "2026-03-08", "usd")
    assert parsed[1] == ("Boba House", "8.10", "03-08-2026", "USD")


def test_normalize_total_cell_handles_symbols_and_locales():
    assert DataReader._normalize_total_cell("$1,234.56") == 1234.56
    assert DataReader._normalize_total_cell("1.234,56") == 1234.56
    assert DataReader._normalize_total_cell("(25.10)") == -25.10


def test_normalize_currency_cell_maps_symbols_and_codes():
    assert DataReader._normalize_currency_cell("$") == "USD"
    assert DataReader._normalize_currency_cell("Paid in eur") == "EUR"
    assert DataReader._normalize_currency_cell("JPY") == "JPY"
    assert DataReader._normalize_currency_cell("") == "USD"


def test_normalize_date_cell_handles_noisy_date_text():
    assert DataReader._normalize_date_cell("Date: 2026-03-08") == "2026-03-08"
    assert DataReader._normalize_date_cell("03/08/2026") == "2026-03-08"


def test_preprocess_data_normalizes_and_drops_invalid_rows():
    data_vector = [
        ("Coffee Shop", "$4.50", "03/08/2026", "$"),
        ("Bad Row", "not-a-number", "2026-03-08", "USD"),
    ]

    processed = DataReader.preprocess_data(data_vector)

    assert len(processed) == 1
    assert processed.iloc[0]["business_name"] == "coffee shop"
    assert processed.iloc[0]["total"] == 4.50
    assert processed.iloc[0]["date"] == "2026-03-08"
    assert processed.iloc[0]["currency"] == "USD"
