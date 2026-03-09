from src.data.data_reader import DataReader
import pandas as pd


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


def test_load_transaction_data_uses_pdf_text_extraction_path(monkeypatch):
    reader = DataReader(config_path="config.conf")
    reader.use_batch_api = False

    monkeypatch.setattr(
        DataReader,
        "gather_files",
        staticmethod(lambda _path: ["/tmp/statement.pdf"]),
    )

    called = {"extract_pdf": 0, "load_proofs": 0}

    monkeypatch.setattr(
        reader,
        "extract_data_from_pdf",
        lambda _pdf_path: called.__setitem__("extract_pdf", called["extract_pdf"] + 1)
        or "[('Coffee Shop', 12.5, '2026-03-08', 'USD')]",
    )
    monkeypatch.setattr(
        reader,
        "load_proofs_data",
        lambda _image_files: called.__setitem__(
            "load_proofs", called["load_proofs"] + 1
        )
        or pd.DataFrame([]),
    )

    result = reader.load_transaction_data("unused")

    assert called["extract_pdf"] == 1
    assert called["load_proofs"] == 0
    assert len(result) == 1
    assert result.iloc[0]["business_name"] == "coffee shop"


def test_load_transaction_data_uses_image_path_for_images(monkeypatch):
    reader = DataReader(config_path="config.conf")

    monkeypatch.setattr(
        DataReader,
        "gather_files",
        staticmethod(lambda _path: ["/tmp/receipt.jpg"]),
    )

    called = {"extract_pdf": 0, "load_proofs": 0}

    monkeypatch.setattr(
        reader,
        "extract_data_from_pdf",
        lambda _pdf_path: called.__setitem__("extract_pdf", called["extract_pdf"] + 1)
        or "[]",
    )
    monkeypatch.setattr(
        reader,
        "load_proofs_data",
        lambda _image_files: called.__setitem__(
            "load_proofs", called["load_proofs"] + 1
        )
        or pd.DataFrame(
            [
                {
                    "business_name": "image store",
                    "total": 10.0,
                    "date": "2026-03-08",
                    "currency": "USD",
                }
            ]
        ),
    )

    result = reader.load_transaction_data("unused")

    assert called["extract_pdf"] == 0
    assert called["load_proofs"] == 1
    assert len(result) == 1
    assert result.iloc[0]["business_name"] == "image store"


def test_extract_data_from_pdf_strips_sensitive_text_before_llm(monkeypatch):
    reader = DataReader(config_path="config.conf")

    monkeypatch.setattr(
        DataReader,
        "_read_pdf_text",
        staticmethod(lambda _pdf_path: "John Doe 4111 1111 1111 1111"),
    )

    captured: dict[str, str] = {}

    def fake_extract(statement_text: str) -> str:
        captured["text"] = statement_text
        return "[]"

    monkeypatch.setattr(reader, "extract_data_from_statement_text", fake_extract)

    reader.extract_data_from_pdf("/tmp/statement.pdf")

    assert "4111" not in captured["text"]
    assert "[CREDIT_CARD]" in captured["text"]
