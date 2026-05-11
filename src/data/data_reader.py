import asyncio
import os
import base64
import logging
import pandas as pd
from time import time
import mimetypes
from PIL import Image
import io
import re
import ast
import json
import random as _random
from typing import Any
from pathlib import Path
from functools import lru_cache

from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from pyhocon import ConfigFactory
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from src.agents.llm_base import LLMBase
from src.prompts.data_reader_prompts import RECEIPT_PROMPT, STATEMENT_PROMPT
from src.utils.currency_conversion_agent import convert_currency_to_usd
from src.data.database import DataBase

logger = logging.getLogger(__name__)


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
        self.fx_max_workers = 8
        # Max concurrent async API calls — controls the asyncio.Semaphore gate
        # for Gemini requests. Keep below your RPM quota (~50-80% of limit).
        self.async_max_concurrent = 30

        super().__init__(
            llm_config_path=llm_config_path,
            config_section="data_ingestion",
            default_temperature=0.0,
            default_top_p=1.0,
            default_max_tokens=350,
        )

        self.primary_model = self.model_name
        self.primary_client = self.init_genai_client()

        # Running totals for cost/usage reporting
        self.ingestion_usage = {
            "model": self.primary_model,
            "input_tokens": 0,
            "output_tokens": 0,
            "llm_calls": 0,
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
        start = time()
        if data_type == DataType.TRANSACTIONS:
            print("\n[Ingestion] Reading Transactions...\n")
            processed_data = self.load_transaction_data(self.transactions_data_path)
        elif data_type == DataType.PROOFS:
            print("\n[Ingestion] Reading Proofs...\n")
            processed_data = self.load_proofs_data(self.proofs_data_path)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        elapsed = round(time() - start, 2)
        summary = self.get_ingestion_cost_summary()
        logger.info(
            "[Ingestion] type=%s | time=%.2fs | model=%s | "
            "input_tokens=%d | output_tokens=%d | "
            "llm_calls=%d | estimated_cost=$%.6f",
            data_type.value,
            elapsed,
            summary["model"],
            summary["inputTokens"],
            summary["outputTokens"],
            summary["llmCalls"],
            summary["estimatedTotalCostUsd"],
        )

        # Keep currency for persistence and conversion pipeline.
        return processed_data

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
        model_name: str,
        is_fallback: bool = False,
    ) -> None:
        """
        Accumulate token counts and estimated cost from a single LLM response.

        Args:
            usage: The usage metadata object attached to the Gemini response, or
                ``None`` if the response did not include usage information.
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

        Uses a thread pool for parallel per-file extraction.

        Args:
            pdf_files: List of absolute paths to PDF files.

        Returns:
            Concatenated normalised DataFrame of all extracted transactions,
            or an empty DataFrame if no files could be processed.
        """
        if not pdf_files:
            return pd.DataFrame([])

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
        Process multiple image payloads concurrently via async Gemini API calls.

        Uses ``asyncio`` with a semaphore-gated concurrency pool and exponential
        backoff with jitter for rate-limit resilience. Falls back to synchronous
        ThreadPoolExecutor when no async-capable client is available.

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

        # Prefer async path when the client supports it (client.aio).
        client = self.primary_client
        has_async = (
            client is not None
            and hasattr(client, "aio")
            and hasattr(client.aio, "models")
        )

        if has_async:
            results = self._run_async_batch(image_payloads)
        else:
            # Fallback: synchronous thread pool for envs without async support.
            logger.info(
                "[Ingestion] Async client unavailable — falling back to ThreadPoolExecutor"
            )
            with ThreadPoolExecutor(
                max_workers=min(self.async_max_concurrent, max(1, len(image_payloads)))
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

    # ------------------------------------------------------------------
    # Async helpers for concurrent Gemini API calls
    # ------------------------------------------------------------------

    def _run_async_batch(self, image_payloads: list[dict]) -> list[str]:
        """
        Bridge between sync callers and the async extraction coroutine.

        Creates a new event loop if none is running, or schedules on the
        existing loop when called from an async context.

        Args:
            image_payloads: Image payloads to process concurrently.

        Returns:
            Ordered list of LLM response strings.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an async context (e.g. Jupyter) — schedule via thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, self._async_batch_extract(image_payloads)
                ).result()
        else:
            return asyncio.run(self._async_batch_extract(image_payloads))

    async def _async_batch_extract(self, image_payloads: list[dict]) -> list[str]:
        """
        Concurrently extract receipt data from all image payloads using async Gemini API.

        A semaphore gates the number of in-flight requests to stay within API
        rate limits. Each request uses exponential backoff with jitter on
        retryable errors (429 rate-limit, 503 service unavailable).

        Args:
            image_payloads: Image payloads to process.

        Returns:
            Ordered list of LLM response strings matching input order.
        """
        semaphore = asyncio.Semaphore(self.async_max_concurrent)
        tasks = [
            self._async_extract_single(payload, semaphore)
            for payload in image_payloads
        ]
        return await asyncio.gather(*tasks)

    async def _async_extract_single(
        self,
        image_payload: dict,
        semaphore: asyncio.Semaphore,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> str:
        """
        Extract data from a single image via the async Gemini client with retry logic.

        Implements exponential backoff with full jitter:
        ``delay = random(0, base_delay * 2^attempt)`` capped at 60 seconds.
        Only retries on transient errors (429, 503, connection errors).

        Args:
            image_payload: Single image payload dict.
            semaphore: Shared semaphore to limit concurrency.
            max_retries: Maximum number of retry attempts before raising.
            base_delay: Initial backoff delay in seconds.

        Returns:
            Raw LLM response string containing receipt data.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": RECEIPT_PROMPT},
                    image_payload,
                ],
            }
        ]

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=min(300, self.max_tokens),
        )
        contents, system_instruction = DataReader._build_gemini_contents(messages)
        if system_instruction:
            config.system_instruction = system_instruction

        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            async with semaphore:
                try:
                    response = await self.primary_client.aio.models.generate_content(
                        model=self.primary_model,
                        contents=contents,
                        config=config,
                    )
                    self._record_usage(
                        getattr(response, "usage_metadata", None),
                        model_name=self.primary_model,
                        is_fallback=False,
                    )
                    return DataReader._response_text(response)

                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    # Only retry on transient / rate-limit errors.
                    is_retryable = any(
                        code in error_str
                        for code in ("429", "503", "rate", "resource_exhausted", "unavailable")
                    )

                    if not is_retryable or attempt >= max_retries:
                        logger.error(
                            "[Async] Non-retryable error or retries exhausted "
                            "(attempt %d/%d): %s",
                            attempt + 1,
                            max_retries + 1,
                            e,
                        )
                        raise RuntimeError(
                            f"Gemini async request failed after {attempt + 1} attempt(s): {e}"
                        ) from e

                    # Exponential backoff with full jitter, capped at 60s.
                    max_delay = min(base_delay * (2 ** attempt), 60.0)
                    jittered_delay = _random.uniform(0, max_delay)
                    logger.warning(
                        "[Async] Retryable error (attempt %d/%d), "
                        "backing off %.2fs: %s",
                        attempt + 1,
                        max_retries + 1,
                        jittered_delay,
                        e,
                    )
                    await asyncio.sleep(jittered_delay)

        # Should not be reached, but guard against it.
        raise RuntimeError(
            f"Gemini async request failed after {max_retries + 1} attempts: {last_error}"
        )

    @staticmethod
    def preprocess_data(data_vector: list) -> pd.DataFrame:
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
