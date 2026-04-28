import os
import base64
import numpy as np
import pandas as pd
from time import time
import mimetypes
from PIL import Image
import io
import re
import ast
import json
from typing import Any
from pathlib import Path
from functools import lru_cache

from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from pyhocon import ConfigFactory
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from src.intelligence.llm_base import LLMBase
from prompts.data_reader_prompts import RECEIPT_PROMPT, STATEMENT_PROMPT
from src.utils.currency_conversion_agent import convert_currency_to_usd
from src.data.database import DataBase


class DataType(Enum):
    TRANSACTIONS = "transactions"
    PROOFS = "proofs"


class DataReader(LLMBase):
    """
    Ingest transaction statements and proof images prior to the validation pipeline.

    Supports PDF bank statements and image receipts. All extraction is routed through
    a native Gemini client. Results are normalised to a four-column DataFrame schema
    (``business_name``, ``total``, ``date``, ``currency``) before being returned.
    """

    def __init__(
        self,
        transactions: list[str] | None = None,
        proofs: list[str] | None = None,
        config_path: str = "config/config.conf",
        llm_config_path: str = "config/llm_config.conf",
        database: DataBase | None = None,
        parsed_config: object | None = None,
    ):
        """
        Initialize the DataReader with optional file paths and config overrides.

        Args:
            transactions: List of file paths to transaction files (PDF or image).
                When provided alongside *proofs*, overrides the config data path.
            proofs: List of file paths to proof image files.
                When provided alongside *transactions*, overrides the config data path.
            config_path: Path to the HOCON app config file. Ignored when
                *parsed_config* is supplied.
            llm_config_path: Path to the HOCON LLM config file.
            database: Optional pre-constructed ``DataBase`` instance. Allows
                dependency injection for tests or shared sessions.
            parsed_config: Pre-parsed config object. Takes precedence over
                *config_path* to avoid redundant disk I/O.
        """
        config = (
            parsed_config
            if parsed_config is not None
            else DataReader._load_config_cached(config_path)
        )
        data_path = config.get("data_path")
        self.transactions_data_path = data_path["transactions"]
        self.proofs_data_path = data_path["proofs"]
        self.validated_data_path = data_path.get("validated", "data/validated")
        self.database = database
        # Concurrency knobs for ingestion performance.
        self.io_max_workers = 8
        self.llm_max_workers = 6
        self.fx_max_workers = 8
        raw_use_batch_api = config.get("llm.use_batch_api", False)
        if isinstance(raw_use_batch_api, str):
            self.use_batch_api = raw_use_batch_api.strip().lower() == "true"
        else:
            self.use_batch_api = bool(raw_use_batch_api)
        self.batch_completion_window = str(
            config.get("llm.batch_completion_window", "24h")
        )
        self.batch_poll_seconds = int(config.get("llm.batch_poll_seconds", 2))
        self.batch_max_wait_seconds = int(config.get("llm.batch_max_wait_seconds", 10))

        super().__init__(
            llm_config_path=llm_config_path,
            config_section="data_ingestion",
            default_temperature=0.0,
            default_top_p=1.0,
            default_max_tokens=350,
        )

        self.primary_llm_provider = "gemini"
        self.primary_model = self.model_name
        # No fallback model is configured for native Gemini mode
        self.fallback_model = self.model_name

        self.primary_client = self.init_genai_client()
        self.fallback_client = None

        # Native Gemini migration: disable legacy OpenAI-style JSONL batch flow.
        self.use_batch_api = False

        # Running totals for cost/usage reporting
        self.ingestion_usage = {
            "model": self.primary_model,
            "input_tokens": 0,
            "output_tokens": 0,
            "llm_calls": 0,
            "batch_runs": 0,
            "standard_runs": 0,
            "fx_calls": 0,
            "fallback_calls": 0,
            "estimated_total_cost_usd": 0.0,
        }

        # Allow callers to supply explicit file lists instead of directory paths
        if transactions and proofs and len(transactions) and len(proofs):
            self.transactions_data_path = transactions
            self.proofs_data_path = proofs

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_config_cached(config_path: str) -> Any:
        """
        Parse and cache the app HOCON config; subsequent calls return the cached result.

        Args:
            config_path: Filesystem path to the ``.conf`` configuration file.

        Returns:
            Parsed config object accessible via dot-notation keys.
        """
        return ConfigFactory.parse_file(config_path)

    def load_data(self, data_type: DataType) -> pd.DataFrame:
        """
        Load and preprocess either transactions or proofs data from the configured paths.

        Args:
            data_type: ``DataType.TRANSACTIONS`` or ``DataType.PROOFS``.

        Returns:
            Normalised DataFrame with columns ``business_name``, ``total``,
            ``date``, and ``currency``.

        Raises:
            ValueError: If *data_type* is not a recognised ``DataType`` value.
        """
        if data_type == DataType.TRANSACTIONS:
            print("\n[Ingestion] Reading Transactions...\n")
            processed_data = self.load_transaction_data(self.transactions_data_path)
        elif data_type == DataType.PROOFS:
            print("\n[Ingestion] Reading Proofs...\n")
            processed_data = self.load_proofs_data(self.proofs_data_path)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        # Keep currency for persistence and conversion pipeline.
        return processed_data

    def _completion_token_kwargs(self, max_tokens: int) -> dict[str, int]:
        """
        Build a ``max_tokens`` kwarg dict capped to the configured ingestion limit.

        Args:
            max_tokens: Requested token budget for this particular call.

        Returns:
            Dict with a single ``max_tokens`` key whose value is
            ``min(max_tokens, self.max_tokens)``.
        """
        return {"max_tokens": min(max_tokens, self.max_tokens)}

    def _sampling_kwargs(self) -> dict[str, float]:
        """
        Return the sampling parameter dict for LLM completion calls.

        Returns:
            Dict with ``temperature`` and ``top_p`` keys sourced from the LLM config.
        """
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    @staticmethod
    def _input_token_rate_per_million(model_name: str) -> float:
        """
        Return the input token cost in USD per one million tokens for a given model.

        Args:
            model_name: Gemini model identifier string.

        Returns:
            Cost per million input tokens in USD. Defaults to 0.10 for unknown models.
        """
        rates = {
            # Keep this table current as pricing evolves.
            "gemini-2.5-flash-lite": 0.10,
        }
        return rates.get(model_name, 0.10)

    @staticmethod
    def _output_token_rate_per_million(model_name: str) -> float:
        """
        Return the output token cost in USD per one million tokens for a given model.

        Args:
            model_name: Gemini model identifier string.

        Returns:
            Cost per million output tokens in USD. Defaults to 0.40 for unknown models.
        """
        rates = {
            "gemini-2.5-flash-lite": 0.40,
        }
        return rates.get(model_name, 0.40)

    def _record_usage(
        self,
        usage: object | None,
        mode: str,
        model_name: str,
        is_fallback: bool = False,
    ) -> None:
        """
        Accumulate token counts and estimated cost from a single LLM response.

        Args:
            usage: The usage metadata object attached to the Gemini response, or
                ``None`` if the response did not include usage information.
            mode: ``"batch"`` or ``"standard"`` — determines which call counter to increment.
            model_name: Model identifier used for per-model cost lookup.
            is_fallback: Whether this call was made on the fallback model path.
        """
        if usage is None:
            return

        # Support both OpenAI-style and Gemini-style attribute names
        input_tokens = int(
            getattr(usage, "prompt_tokens", None)
            or getattr(usage, "prompt_token_count", 0)
            or 0
        )
        output_tokens = int(
            getattr(usage, "completion_tokens", None)
            or getattr(usage, "candidates_token_count", 0)
            or 0
        )
        self.ingestion_usage["input_tokens"] += input_tokens
        self.ingestion_usage["output_tokens"] += output_tokens
        self.ingestion_usage["llm_calls"] += 1

        if mode == "batch":
            self.ingestion_usage["batch_runs"] += 1
        else:
            self.ingestion_usage["standard_runs"] += 1

        if is_fallback:
            self.ingestion_usage["fallback_calls"] += 1

        input_cost = (
            input_tokens / 1_000_000
        ) * DataReader._input_token_rate_per_million(model_name)
        output_cost = (
            output_tokens / 1_000_000
        ) * DataReader._output_token_rate_per_million(model_name)
        self.ingestion_usage["estimated_total_cost_usd"] += input_cost + output_cost

    def _record_usage_from_body(self, body: dict, mode: str, model_name: str) -> None:
        """
        Accumulate token usage from a raw response body dict (legacy batch path).

        Args:
            body: Parsed JSON response body dict; expects an ``"usage"`` key.
            mode: ``"batch"`` or ``"standard"``.
            model_name: Model identifier used for per-model cost lookup.
        """
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)

        self.ingestion_usage["input_tokens"] += input_tokens
        self.ingestion_usage["output_tokens"] += output_tokens
        self.ingestion_usage["llm_calls"] += 1

        if mode == "batch":
            self.ingestion_usage["batch_runs"] += 1
        else:
            self.ingestion_usage["standard_runs"] += 1

        input_cost = (
            input_tokens / 1_000_000
        ) * DataReader._input_token_rate_per_million(model_name)
        output_cost = (
            output_tokens / 1_000_000
        ) * DataReader._output_token_rate_per_million(model_name)
        self.ingestion_usage["estimated_total_cost_usd"] += input_cost + output_cost

    def get_ingestion_cost_summary(self) -> dict[str, Any]:
        """Return normalized token/cost metrics for the current ingestion run."""
        model_name = self.ingestion_usage["model"]
        input_tokens = int(self.ingestion_usage["input_tokens"])
        output_tokens = int(self.ingestion_usage["output_tokens"])

        input_cost = (
            input_tokens / 1_000_000
        ) * DataReader._input_token_rate_per_million(model_name)
        output_cost = (
            output_tokens / 1_000_000
        ) * DataReader._output_token_rate_per_million(model_name)

        return {
            "model": model_name,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "llmCalls": int(self.ingestion_usage["llm_calls"]),
            "batchCalls": int(self.ingestion_usage["batch_runs"]),
            "standardCalls": int(self.ingestion_usage["standard_runs"]),
            "fxCalls": int(self.ingestion_usage["fx_calls"]),
            "fallbackCalls": int(self.ingestion_usage["fallback_calls"]),
            "estimatedInputCostUsd": round(input_cost, 2),
            "estimatedOutputCostUsd": round(output_cost, 2),
            "estimatedTotalCostUsd": round(
                float(self.ingestion_usage["estimated_total_cost_usd"]), 2
            ),
        }

    def log_ingestion_cost(self, session_id: str) -> dict:
        """
        Append the current ingestion cost summary to the session log file and return it.

        Args:
            session_id: External session identifier included in the log entry.

        Returns:
            The cost summary dict as returned by ``get_ingestion_cost_summary()``.
        """
        summary = self.get_ingestion_cost_summary()
        log_entry = {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "sessionId": session_id,
            "ingestion": summary,
        }

        os.makedirs(self.validated_data_path, exist_ok=True)
        log_path = os.path.join(self.validated_data_path, "ingestion_cost.log")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")

        print(f"\nIngestion usage: {json.dumps(log_entry)}\n")
        return summary

    @staticmethod
    def _extract_text_content(message_content: object) -> str:
        """
        Extract a plain-text string from a message content value.

        Handles string content, lists of text/dict items (OpenAI-style multi-part
        content), and arbitrary objects by falling back to ``str()``.

        Args:
            message_content: Raw content from a message dict.

        Returns:
            Concatenated text extracted from the content.
        """
        if isinstance(message_content, str):
            return message_content

        if isinstance(message_content, list):
            parts: list[str] = []
            for item in message_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)

        return str(message_content)

    def _poll_batch_until_done(self, batch_id: str):
        """
        Poll a batch job until it completes (disabled for native Gemini mode).

        Args:
            batch_id: The batch job identifier to poll.

        Raises:
            RuntimeError: Always — batch API is disabled in native Gemini mode.
        """
        _ = batch_id
        raise RuntimeError("Batch API is disabled for native Gemini ingestion mode.")

    @staticmethod
    def _serialize_batch_output(raw_content: object) -> str:
        """
        Convert a raw batch response object to a plain string.

        Tries ``text``, ``read()``, and ``content`` attributes in order before
        falling back to ``str()``.

        Args:
            raw_content: The raw response object from a batch API call.

        Returns:
            A UTF-8 string representation of the response content.
        """
        if hasattr(raw_content, "text"):
            return str(raw_content.text)
        if hasattr(raw_content, "read"):
            read_val = raw_content.read()
            if isinstance(read_val, bytes):
                return read_val.decode("utf-8")
            return str(read_val)
        if hasattr(raw_content, "content"):
            content = raw_content.content
            if isinstance(content, bytes):
                return content.decode("utf-8")
            return str(content)
        return str(raw_content)

    def _run_chat_batch_requests(
        self, requests_payload: list[dict], model_name: str
    ) -> list[str]:
        """
        Execute a batch of chat completion requests (disabled for native Gemini mode).

        Args:
            requests_payload: List of request dicts to submit as a batch.
            model_name: Model identifier to use for the batch job.

        Raises:
            RuntimeError: Always — batch API is disabled in native Gemini mode.
        """
        _ = requests_payload
        _ = model_name
        raise RuntimeError("Batch API is disabled for native Gemini ingestion mode.")

    def load_proofs_data(self, data_path: str | list[str]) -> pd.DataFrame:
        """
        Extract receipt data from image files at *data_path*.

        Images are encoded to base64, sent to the Gemini vision API concurrently,
        and then normalised into a four-column DataFrame. Non-USD totals are
        converted to USD via the exchange-rate API.

        Args:
            data_path: Either a directory path (str) or a list of image file paths.

        Returns:
            Normalised DataFrame with columns ``business_name``, ``total``,
            ``date``, and ``currency`` (all in USD after conversion).
        """
        print("\n[Ingestion] Starting proof extraction\n")
        start = time()
        payload = DataReader.create_image_payload(data_path, self.io_max_workers)
        data = self.batch_read_data(payload)
        end = time()
        print(f"\nTime to read proofs: {round(end - start, 2)}s\n")

        data_vec = []
        for item in data:
            parsed = ast.literal_eval(item)
            data_vec.extend(parsed)

        processed_data = DataReader.preprocess_data(data_vec)

        # Convert currency to USD if necessary
        if (processed_data["currency"] != "USD").any():
            non_usd_data = processed_data[processed_data["currency"] != "USD"]
            entries = non_usd_data.to_dict(orient="records")
            self.ingestion_usage["fx_calls"] += len(entries)
            with ThreadPoolExecutor(
                max_workers=min(self.fx_max_workers, len(entries))
            ) as executor:
                converted = list(executor.map(convert_currency_to_usd, entries))
            processed_data.loc[non_usd_data.index, "total"] = converted
            processed_data.loc[non_usd_data.index, "currency"] = "USD"

        return processed_data

    @staticmethod
    def strip_sensitive_info(text):
        """
        Redact personally identifiable information from statement text before LLM submission.

        Replaces emails, phone numbers, dates, zip codes, credit card numbers,
        account numbers, addresses, and full names with placeholder tokens.

        Args:
            text: Raw text extracted from a bank/card statement PDF.

        Returns:
            The input text with PII replaced by ``[EMAIL]``, ``[PHONE]``, etc.
        """
        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", text)
        text = re.sub(
            r"\b(?:\+?1[-.\s]?)*\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "[PHONE]",
            text,
        )
        text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE]", text)
        text = re.sub(r"\b\d{5}(?:-\d{4})?\b", "[ZIP]", text)
        text = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[CREDIT_CARD]", text)
        text = re.sub(
            r"\b(?:Account|Acct|A/C)[\s#:]*\d{8,20}\b",
            "[ACCOUNT]",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\b\d{8,20}\b", "[ACCOUNT]", text)
        text = re.sub(
            r"\d{1,5}\s+\w+(\s+\w+)*\s+(Street|St|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane)\b",
            "[ADDRESS]",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\b([A-Z][a-z]+\s[A-Z][a-z]+)\b", "[NAME]", text)

        return text

    def load_transaction_data(self, data_path: str | list[str]) -> pd.DataFrame:
        """
        Load transaction data from PDF statements and/or image files.

        PDF and image files are detected automatically. When both are present they
        are processed concurrently and their results concatenated. PII is stripped
        from PDF text before it is sent to the LLM.

        Args:
            data_path: Either a directory path (str) or a list of file paths.

        Returns:
            Normalised DataFrame with columns ``business_name``, ``total``,
            ``date``, and ``currency``.
        """
        print("\n[Ingestion] Starting transaction extraction\n")
        start = time()
        data = pd.DataFrame([])

        all_files = DataReader.gather_files(data_path)
        pdf_files = [f for f in all_files if Path(f).suffix.lower() == ".pdf"]
        image_files = [
            f for f in all_files if Path(f).suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]

        if pdf_files and image_files:
            # PDF text extraction and image extraction are independent; run in parallel.
            with ThreadPoolExecutor(max_workers=2) as executor:
                pdf_future = executor.submit(self._load_pdf_transaction_data, pdf_files)
                image_future = executor.submit(
                    self._load_image_transaction_data, image_files
                )
                pdf_data = pdf_future.result()
                image_data = image_future.result()

            frames = [frame for frame in [pdf_data, image_data] if not frame.empty]
            if frames:
                data = pd.concat(frames, axis=0, ignore_index=True)
        elif pdf_files:
            data = self._load_pdf_transaction_data(pdf_files)
        elif image_files:
            data = self._load_image_transaction_data(image_files)

        end = time()
        print(f"\nTime to read transaction statements: {round(end - start, 2)}s\n")

        return data

    def _load_pdf_transaction_data(self, pdf_files: list[str]) -> pd.DataFrame:
        """
        Extract transactions from a list of PDF statement files.

        Attempts the standard per-file extraction path (parallel via thread pool).
        The legacy batch path is retained but will only run when ``use_batch_api``
        is True, which is currently always False in native Gemini mode.

        Args:
            pdf_files: List of absolute paths to PDF files.

        Returns:
            Concatenated normalised DataFrame of all extracted transactions,
            or an empty DataFrame if no files could be processed.
        """
        if not pdf_files:
            return pd.DataFrame([])

        pdf_frames: list[pd.DataFrame] = []

        if self.use_batch_api:
            try:
                statement_texts = [
                    DataReader.strip_sensitive_info(self._read_pdf_text(path))
                    for path in pdf_files
                ]
                extracted = self.extract_statement_data_batch(statement_texts)
                for item in extracted:
                    if not item:
                        continue
                    data_vec = ast.literal_eval(item)
                    pdf_frames.append(DataReader.preprocess_data(data_vec))
            except Exception as e:
                print(
                    f"\nWarning: Batch PDF extraction failed; falling back. Error: {e}\n"
                )

        if not pdf_frames:

            def process_pdf(pdf_path: str) -> pd.DataFrame:
                try:
                    extracted_data = self.extract_data_from_pdf(pdf_path)
                    data_vec = ast.literal_eval(extracted_data)
                    return DataReader.preprocess_data(data_vec)
                except Exception as e:
                    print(f"\nWarning: Failed to process PDF {pdf_path}: {e}\n")
                    return pd.DataFrame([])

            with ThreadPoolExecutor(
                max_workers=min(self.io_max_workers, len(pdf_files))
            ) as executor:
                pdf_frames = list(executor.map(process_pdf, pdf_files))

        valid_pdf_frames = [frame for frame in pdf_frames if not frame.empty]
        if not valid_pdf_frames:
            return pd.DataFrame([])

        return pd.concat(valid_pdf_frames, axis=0, ignore_index=True)

    def _load_image_transaction_data(self, image_files: list[str]) -> pd.DataFrame:
        """
        Extract transactions from image files by delegating to the proof extraction path.

        Args:
            image_files: List of absolute paths to image files (.png, .jpg, .jpeg).

        Returns:
            Normalised DataFrame of extracted transactions, or an empty DataFrame on failure.
        """
        if not image_files:
            return pd.DataFrame([])

        try:
            return self.load_proofs_data(image_files)
        except Exception as e:
            print(f"\nWarning: Failed to process image files: {e}\n")
            return pd.DataFrame([])

    @staticmethod
    def gather_files(data_path: str | list[str]) -> list[str]:
        """
        Resolve a data path to a flat list of file paths.

        Args:
            data_path: A single file path, a directory path, or a list of file paths.

        Returns:
            List of resolved file path strings.

        Raises:
            ValueError: If *data_path* is neither a string path nor a list.
        """
        if isinstance(data_path, str):
            p = Path(data_path)
            if p.is_file():
                return [str(p)]
            if p.is_dir():
                return [str(f) for f in p.iterdir() if f.is_file()]
        elif isinstance(data_path, list):
            return [str(file_path) for file_path in data_path]

        raise ValueError(
            "data_path must be a file path, directory path, or list of file paths."
        )

    def extract_data_from_pdf(self, pdf_path: str) -> str:
        """
        Extract structured transaction data from a single PDF file.

        Reads raw text from the PDF, strips PII, then sends the sanitised text
        to the statement extraction LLM prompt.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            Raw LLM response string (Python list literal) containing extracted rows.
        """
        filtered_statement_text = DataReader.strip_sensitive_info(
            self._read_pdf_text(pdf_path)
        )
        data = self.extract_data_from_statement_text(filtered_statement_text)

        return data

    @staticmethod
    def _read_pdf_text(pdf_path: str) -> str:
        """
        Concatenate the raw text content of all pages in a PDF file.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            Plain text string with all page contents joined together.
        """
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text = ""
        for doc in docs:
            text += doc.page_content
        return text

    def batch_read_data(self, image_payloads: list[dict]) -> list[str]:
        """
        Process multiple image payloads concurrently and return extracted text responses.

        Attempts the batch API path first (currently disabled) and falls back to
        per-image parallel extraction via a thread pool.

        Args:
            image_payloads: List of image payload dicts as produced by
                ``create_image_payload()``.

        Returns:
            List of raw LLM response strings, one per input payload.
        """
        if not image_payloads:
            return []

        start_time = time()
        start_cost = float(self.ingestion_usage["estimated_total_cost_usd"])
        start_llm_calls = int(self.ingestion_usage["llm_calls"])
        start_fallback_calls = int(self.ingestion_usage["fallback_calls"])

        can_use_batch_api = self.use_batch_api and self.primary_client is not None
        if can_use_batch_api and image_payloads:
            try:
                results = self.read_proofs_data_batch(image_payloads)
                elapsed = time() - start_time
                cost_delta = (
                    float(self.ingestion_usage["estimated_total_cost_usd"]) - start_cost
                )
                llm_calls_delta = (
                    int(self.ingestion_usage["llm_calls"]) - start_llm_calls
                )
                fallback_calls_delta = (
                    int(self.ingestion_usage["fallback_calls"]) - start_fallback_calls
                )
                avg_latency = elapsed / max(1, len(image_payloads))
                print(
                    "\n[Ingestion] Image extraction summary: "
                    f"images={len(image_payloads)}, "
                    f"total_latency_s={elapsed:.2f}, "
                    f"avg_latency_s={avg_latency:.2f}, "
                    f"llm_calls={llm_calls_delta}, "
                    f"fallback_calls={fallback_calls_delta}, "
                    f"estimated_cost_usd={cost_delta:.6f}\n"
                )
                return results
            except Exception as e:
                print(
                    f"\nWarning: Batch proof extraction failed; falling back. Error: {e}\n"
                )

        with ThreadPoolExecutor(
            max_workers=min(self.llm_max_workers, max(1, len(image_payloads)))
        ) as executor:
            results = list(executor.map(self.read_proofs_data, image_payloads))

        elapsed = time() - start_time
        cost_delta = (
            float(self.ingestion_usage["estimated_total_cost_usd"]) - start_cost
        )
        llm_calls_delta = int(self.ingestion_usage["llm_calls"]) - start_llm_calls
        fallback_calls_delta = (
            int(self.ingestion_usage["fallback_calls"]) - start_fallback_calls
        )
        avg_latency = elapsed / max(1, len(image_payloads))
        print(
            "\n[Ingestion] Image extraction summary: "
            f"images={len(image_payloads)}, "
            f"total_latency_s={elapsed:.2f}, "
            f"avg_latency_s={avg_latency:.2f}, "
            f"llm_calls={llm_calls_delta}, "
            f"fallback_calls={fallback_calls_delta}, "
            f"estimated_cost_usd={cost_delta:.6f}\n"
        )

        return results

    @staticmethod
    def preprocess_data(data_vector: np.ndarray) -> pd.DataFrame:
        """
        Normalise a raw extraction result into the canonical four-column DataFrame.

        Handles both 3-tuple rows (name, total, date) from PDF statements and
        4-tuple rows (name, total, date, currency) from image receipts.

        Args:
            data_vector: List or array of row tuples/lists as returned by the LLM.

        Returns:
            A DataFrame with columns ``business_name``, ``total``, ``date``,
            and ``currency``. Business names are lowercased; totals are cast to float.

        Raises:
            ValueError: If rows contain fewer than three fields.
        """
        expected_cols = ["business_name", "total", "date", "currency"]
        data = pd.DataFrame(data_vector)

        # Statement PDFs can produce 3-tuples (name, total, date).
        # Normalize to the canonical 4-column schema by defaulting currency to USD.
        if data.empty:
            data = pd.DataFrame([], columns=expected_cols)
        elif data.shape[1] == 3:
            data.columns = expected_cols[:3]
            data["currency"] = "USD"
        elif data.shape[1] >= 4:
            data = data.iloc[:, :4]
            data.columns = expected_cols
        else:
            raise ValueError(
                "Extracted data must contain at least business_name, total, and date."
            )

        data["business_name"] = data["business_name"].astype(str).str.lower()
        data["total"] = data["total"].astype(float)

        return data

    @staticmethod
    def create_image_payload(
        data_path: str | list[str], max_workers: int = 8
    ) -> list[dict]:
        """
        Build a list of base64-encoded image payload dicts ready for the LLM.

        Only files whose MIME type starts with ``image/`` are included. Encoding
        is done in parallel using a thread pool.

        Args:
            data_path: Directory path (str) or list of absolute file paths.
            max_workers: Maximum number of threads for concurrent encoding.

        Returns:
            List of ``{"type": "image_url", "image_url": {"url": "data:..."}}`` dicts.
        """
        image_payload = []
        is_dir = isinstance(data_path, str)
        files = os.listdir(data_path) if is_dir else data_path

        def process_file(file_name):
            mime_type, _ = mimetypes.guess_type(file_name)
            if mime_type and mime_type.startswith("image/"):
                image_path = os.path.join(data_path, file_name) if is_dir else file_name
                encoded_string = DataReader.encode_image(image_path)
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_string}",
                    },
                }

            return None

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            results = list(executor.map(process_file, files))

        image_payload = [result for result in results if result is not None]
        return image_payload

    @staticmethod
    def reduce_image_size(image_path, max_size=2 * 1024 * 1024):
        """
        Compress an image in-memory until it falls below *max_size* bytes.

        Iteratively reduces JPEG quality by 5 points per pass, then falls back to
        resizing by 5% per pass if quality reduction alone is insufficient.
        The image is never written to disk.

        Args:
            image_path: Path to the source image file.
            max_size: Maximum acceptable file size in bytes. Defaults to 2 MB.

        Returns:
            ``io.BytesIO`` buffer positioned at byte 0 containing the compressed image.
        """
        img = Image.open(image_path)

        # Convert palette/alpha modes to RGB for JPEG compatibility
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        img_bytes = io.BytesIO()

        quality = 90
        while True:
            img_bytes.seek(0)
            img_bytes.truncate(0)
            img.save(img_bytes, format="JPEG", quality=quality)

            if img_bytes.tell() <= max_size or quality <= 10:
                break

            quality -= 5

        while img_bytes.tell() > max_size:
            width, height = img.size
            img = img.resize((int(width * 0.95), int(height * 0.95)), Image.LANCZOS)
            img_bytes.seek(0)
            img_bytes.truncate(0)
            img.save(img_bytes, format="JPEG", quality=quality)

        img_bytes.seek(0)
        return img_bytes

    @staticmethod
    def encode_image(image_path: str):
        """
        Encode an image file to a base64 string after size reduction.

        Args:
            image_path: Path to the source image file.

        Returns:
            Base64-encoded UTF-8 string of the (possibly compressed) image.
        """
        img_bytes = DataReader.reduce_image_size(image_path)
        return base64.b64encode(img_bytes.read()).decode("utf-8")

    @staticmethod
    def _parse_data_url_image(url: str) -> tuple[bytes, str]:
        """
        Decode a data-URL image string into raw bytes and its MIME type.

        Args:
            url: Data URL string of the form ``data:<mime>;base64,<data>``.

        Returns:
            A tuple ``(image_bytes, mime_type)``.

        Raises:
            ValueError: If *url* does not start with ``"data:"``.
        """
        if not isinstance(url, str) or not url.startswith("data:"):
            raise ValueError("Unsupported image URL payload for Gemini request.")

        header, encoded = url.split(",", 1)
        mime_type = header.split(";", 1)[0].replace("data:", "", 1)
        return base64.b64decode(encoded), mime_type

    @staticmethod
    def _build_gemini_contents(messages: list[dict]) -> tuple[list[types.Part], str]:
        """
        Convert an OpenAI-style message list into Gemini ``Part`` objects and a system prompt.

        System messages are collected into a single string returned as the second
        element. User messages are converted to ``Part.from_text`` or
        ``Part.from_bytes`` (for image_url items).

        Args:
            messages: List of ``{"role": ..., "content": ...}`` message dicts.

        Returns:
            A tuple ``(parts, system_instruction)`` where *parts* is a list of
            ``types.Part`` objects and *system_instruction* is the merged system text.
        """
        parts: list[types.Part] = []
        system_texts: list[str] = []

        for msg in messages:
            role = str(msg.get("role", "")).strip().lower()
            content = msg.get("content", "")

            if role == "system":
                # Collect system instructions separately for the system_instruction kwarg
                text = DataReader._extract_text_content(content).strip()
                if text:
                    system_texts.append(text)
                continue

            if isinstance(content, str):
                if content.strip():
                    parts.append(types.Part.from_text(text=content))
                continue

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        if item.strip():
                            parts.append(types.Part.from_text(text=item))
                        continue

                    if not isinstance(item, dict):
                        continue

                    if item.get("type") == "text":
                        text_val = str(item.get("text", "") or "").strip()
                        if text_val:
                            parts.append(types.Part.from_text(text=text_val))
                    elif item.get("type") == "image_url":
                        img_url = ((item.get("image_url") or {}).get("url")) or ""
                        image_bytes, mime_type = DataReader._parse_data_url_image(
                            str(img_url)
                        )
                        parts.append(
                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type=mime_type,
                            )
                        )

        return parts, "\n".join(system_texts).strip()

    @staticmethod
    def _response_text(response: object) -> str:
        """
        Extract the text string from a Gemini ``GenerateContentResponse``.

        Tries the top-level ``text`` attribute first, then iterates over candidates
        and their parts to find any non-empty text chunk.

        Args:
            response: A Gemini ``GenerateContentResponse`` object.

        Returns:
            The first non-empty text found, or an empty string if none is present.
        """
        text = str(getattr(response, "text", "") or "").strip()
        if text:
            return text

        # Fall back to iterating candidates when the top-level text attribute is empty
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            cparts = getattr(content, "parts", None) if content is not None else None
            if not cparts:
                continue
            chunk = []
            for part in cparts:
                part_text = str(getattr(part, "text", "") or "")
                if part_text:
                    chunk.append(part_text)
            merged = "".join(chunk).strip()
            if merged:
                return merged

        return ""

    def read_proofs_data(self, image_payload: dict) -> str:
        """
        Send a single encoded image payload to the Gemini vision API and return the extracted text.

        Args:
            image_payload: A single image payload dict produced by ``create_image_payload()``.

        Returns:
            Raw LLM response string (Python list literal) containing receipt data.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": RECEIPT_PROMPT,
                    },
                    image_payload,
                ],
            }
        ]
        return self._chat_completion_with_fallback(messages, max_tokens=300)

    def _chat_completion_with_fallback(
        self, messages: list[dict], max_tokens: int
    ) -> str:
        """
        Submit a message list to the primary Gemini model and return the text response.

        Builds ``GenerateContentConfig`` from the current sampling parameters, converts
        the OpenAI-style message list to Gemini ``Part`` objects, and records usage.
        Raises ``RuntimeError`` if the primary client is unavailable or the request fails.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` message dicts.
            max_tokens: Maximum number of output tokens requested for this call.

        Returns:
            Text response string from the model.

        Raises:
            RuntimeError: If the Gemini request fails and no fallback is available.
        """
        completion_kwargs = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=min(max_tokens, self.max_tokens),
        )

        contents, system_instruction = DataReader._build_gemini_contents(messages)
        # Attach system instruction when present; not all prompts include one
        if system_instruction:
            completion_kwargs.system_instruction = system_instruction

        if self.primary_client is not None:
            try:
                response = self.primary_client.models.generate_content(
                    model=self.primary_model,
                    contents=contents,
                    config=completion_kwargs,
                )
                self._record_usage(
                    getattr(response, "usage_metadata", None),
                    mode="standard",
                    model_name=self.primary_model,
                    is_fallback=False,
                )
                return DataReader._response_text(response)
            except Exception as e:
                print(
                    f"\nWarning: Gemini model {self.primary_model} request failed. "
                    f"Error: {e}\n"
                )

        raise RuntimeError(
            f"Gemini request failed for model {self.primary_model} and no fallback is enabled."
        )

    def read_proofs_data_batch(self, image_payloads: list[dict]) -> list[str]:
        """
        Batch extraction endpoint for proof images (disabled for native Gemini mode).

        Args:
            image_payloads: List of image payload dicts.

        Raises:
            RuntimeError: Always — batch API is disabled in native Gemini mode.
        """
        _ = image_payloads
        raise RuntimeError("Batch API is disabled for native Gemini ingestion mode.")

    def extract_data_from_statement_text(self, bank_statement_text: str) -> str:
        """
        Extract structured transaction rows from sanitised bank statement text.

        Args:
            bank_statement_text: PII-stripped plain text from a bank/card statement.

        Returns:
            Raw LLM response string (Python list literal) containing extracted rows.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": STATEMENT_PROMPT + "\n\n" + bank_statement_text,
            },
        ]
        return self._chat_completion_with_fallback(messages, max_tokens=350)

    def extract_data_from_image_texts(self, bank_statement_text: str) -> str:
        """
        Backward-compatible alias for ``extract_data_from_statement_text``.

        Args:
            bank_statement_text: PII-stripped plain text from a bank/card statement.

        Returns:
            Raw LLM response string (Python list literal) containing extracted rows.
        """
        return self.extract_data_from_statement_text(bank_statement_text)

    def extract_statement_data_batch(self, statement_texts: list[str]) -> list[str]:
        """
        Batch statement extraction endpoint (disabled for native Gemini mode).

        Args:
            statement_texts: List of PII-stripped statement text strings.

        Raises:
            RuntimeError: Always — batch API is disabled in native Gemini mode.
        """
        _ = statement_texts
        raise RuntimeError("Batch API is disabled for native Gemini ingestion mode.")
