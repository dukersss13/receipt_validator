import os
import base64
import numpy as np
import pandas as pd
from time import time
import time as wall_time
import mimetypes
from PIL import Image
import io
import re
import ast
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from functools import lru_cache

from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from pyhocon import ConfigFactory
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from src.prompts import RECEIPT_PROMPT, STATEMENT_PROMPT
from src.utils.currency_conversion_agent import convert_currency_to_usd
from src.utils.utils import (
    setup_openai,
    setup_gemini,
    OPENAI_MODEL,
    GEMINI_FLASH_LITE_MODEL,
)
from src.data.database import DataBase


class DataType(Enum):
    TRANSACTIONS = "transactions"
    PROOFS = "proofs"


class DataReader:
    """
    Data Reader class ingests transactions and proofs images
    prior to the validation process.
    """

    def __init__(
        self,
        transactions: list[str] | None = None,
        proofs: list[str] | None = None,
        config_path: str = "config.conf",
        database: DataBase | None = None,
        parsed_config: object | None = None,
    ):
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

        self.primary_llm_provider = (
            str(config.get("llm.default_provider", "gemini_flash_lite")).strip().lower()
        )
        self.primary_model = str(
            config.get("llm.primary_model", GEMINI_FLASH_LITE_MODEL)
        )
        self.fallback_model = str(config.get("llm.fallback_model", OPENAI_MODEL))
        self.gemini_base_url = str(
            config.get(
                "llm.gemini_base_url",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        )

        setup_openai()
        self.openai_client = OpenAI()

        setup_gemini()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_client = (
            OpenAI(api_key=gemini_key, base_url=self.gemini_base_url)
            if gemini_key
            else None
        )

        if self.primary_llm_provider == "openai":
            self.primary_client = self.openai_client
            self.primary_model = self.fallback_model
            self.fallback_client = self.gemini_client
            self.fallback_model = GEMINI_FLASH_LITE_MODEL
        else:
            self.primary_client = self.gemini_client
            self.fallback_client = self.openai_client

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

        if transactions and proofs and len(transactions) and len(proofs):
            self.transactions_data_path = transactions
            self.proofs_data_path = proofs

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_config_cached(config_path: str):
        """Cache parsed config to avoid repeated disk parsing per request."""
        return ConfigFactory.parse_file(config_path)

    def load_data(self, data_type: DataType):
        """
        Read (preprocess) data given data type

        :param data_type: type of data, transactions or proofs
        :return: data in a df format
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

    @staticmethod
    def _completion_token_kwargs(max_tokens: int) -> dict:
        return {"max_tokens": max_tokens}

    @staticmethod
    def _sampling_kwargs() -> dict:
        return {"temperature": 0.0}

    @staticmethod
    def _input_token_rate_per_million(model_name: str) -> float:
        rates = {
            # Keep this table current as pricing evolves.
            "gpt-4o-mini": 0.15,
            "gemini-2.5-flash-lite": 0.10,
        }
        return rates.get(model_name, 0.15)

    @staticmethod
    def _output_token_rate_per_million(model_name: str) -> float:
        rates = {
            "gpt-4o-mini": 0.60,
            "gemini-2.5-flash-lite": 0.40,
        }
        return rates.get(model_name, 0.60)

    def _record_usage(
        self,
        usage: object | None,
        mode: str,
        model_name: str,
        is_fallback: bool = False,
    ) -> None:
        if usage is None:
            return

        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
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

    def get_ingestion_cost_summary(self) -> dict:
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
        started = wall_time.time()
        terminal = {"completed", "failed", "cancelled", "expired"}

        while True:
            batch = self.openai_client.batches.retrieve(batch_id)
            if batch.status in terminal:
                return batch

            if (wall_time.time() - started) > self.batch_max_wait_seconds:
                raise TimeoutError(
                    f"Batch {batch_id} did not finish within {self.batch_max_wait_seconds}s"
                )

            wall_time.sleep(max(1, self.batch_poll_seconds))

    @staticmethod
    def _serialize_batch_output(raw_content: object) -> str:
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
        if not requests_payload:
            return []

        if not hasattr(self.openai_client, "batches"):
            raise RuntimeError("OpenAI client does not support Batch API.")

        with NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp:
            for item in requests_payload:
                tmp.write(json.dumps(item) + "\n")
            input_file_path = tmp.name

        output_text: str | None = None
        try:
            with open(input_file_path, "rb") as f_obj:
                input_file = self.openai_client.files.create(
                    file=f_obj, purpose="batch"
                )

            batch = self.openai_client.batches.create(
                input_file_id=input_file.id,
                endpoint="/v1/chat/completions",
                completion_window=self.batch_completion_window,
            )

            completed_batch = self._poll_batch_until_done(batch.id)
            if (
                completed_batch.status != "completed"
                or not completed_batch.output_file_id
            ):
                raise RuntimeError(
                    f"Batch failed with status {completed_batch.status}."
                )

            output = self.openai_client.files.content(completed_batch.output_file_id)
            output_text = DataReader._serialize_batch_output(output)
        finally:
            try:
                os.remove(input_file_path)
            except OSError:
                pass

        lines = (
            [line for line in output_text.splitlines() if line.strip()]
            if output_text
            else []
        )
        by_id: dict[str, str] = {}

        for line in lines:
            obj = json.loads(line)
            custom_id = str(obj.get("custom_id", ""))
            body = ((obj.get("response") or {}).get("body")) or {}
            self._record_usage_from_body(body, mode="batch", model_name=model_name)

            choices = body.get("choices", []) if isinstance(body, dict) else []
            if not choices:
                by_id[custom_id] = ""
                continue

            msg_content = (choices[0].get("message") or {}).get("content", "")
            by_id[custom_id] = DataReader._extract_text_content(msg_content)

        outputs: list[str] = []
        for req in requests_payload:
            outputs.append(by_id.get(str(req.get("custom_id", "")), ""))

        return outputs

    def load_proofs_data(self, data_path: str | list[str]) -> pd.DataFrame:
        """
        Loads proof data from the specified path.
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
        Loads transaction data from PDFs and image files in a given path or list of file paths.
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
        """Read transactions from PDF statements and normalize them into one frame."""
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
        """Read transactions from uploaded images using the proof-image extraction path."""
        if not image_files:
            return pd.DataFrame([])

        try:
            return self.load_proofs_data(image_files)
        except Exception as e:
            print(f"\nWarning: Failed to process image files: {e}\n")
            return pd.DataFrame([])

    @staticmethod
    def gather_files(data_path: str | list[str]) -> list[str]:
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
        Extracts data from a PDF file located at the specified path.
        """
        filtered_statement_text = DataReader.strip_sensitive_info(
            self._read_pdf_text(pdf_path)
        )
        data = self.extract_data_from_statement_text(filtered_statement_text)

        return data

    @staticmethod
    def _read_pdf_text(pdf_path: str) -> str:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text = ""
        for doc in docs:
            text += doc.page_content
        return text

    def batch_read_data(self, image_payloads: list[dict]) -> list[str]:
        """
        Process multiple image payloads in parallel.
        """
        if not image_payloads:
            return []

        start_time = time()
        start_cost = float(self.ingestion_usage["estimated_total_cost_usd"])
        start_llm_calls = int(self.ingestion_usage["llm_calls"])
        start_fallback_calls = int(self.ingestion_usage["fallback_calls"])

        can_use_openai_batch = (
            self.use_batch_api and self.primary_llm_provider == "openai"
        )
        if can_use_openai_batch and image_payloads:
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
        Preprocess data post ingestion.
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
        Create image payload.
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
        Reduce an image's size to be under max_size (default 2MB) without saving to disk.
        """
        img = Image.open(image_path)

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
        Function to encode the image.
        """
        img_bytes = DataReader.reduce_image_size(image_path)
        return base64.b64encode(img_bytes.read()).decode("utf-8")

    def read_proofs_data(self, image_payload: dict) -> str:
        """
        Read proof image payload and extract receipt information.
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
        completion_kwargs = {
            **DataReader._sampling_kwargs(),
            **DataReader._completion_token_kwargs(max_tokens),
        }

        if self.primary_client is not None:
            try:
                response = self.primary_client.chat.completions.create(
                    model=self.primary_model,
                    messages=messages,
                    **completion_kwargs,
                )
                self._record_usage(
                    response.usage,
                    mode="standard",
                    model_name=self.primary_model,
                    is_fallback=False,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(
                    f"\nWarning: Primary model {self.primary_model} failed; "
                    f"falling back to {self.fallback_model}. Error: {e}\n"
                )

        if self.fallback_client is None:
            raise RuntimeError(
                f"No fallback client configured for model {self.fallback_model}."
            )

        response = self.fallback_client.chat.completions.create(
            model=self.fallback_model,
            messages=messages,
            **completion_kwargs,
        )
        self._record_usage(
            response.usage,
            mode="standard",
            model_name=self.fallback_model,
            is_fallback=True,
        )
        return response.choices[0].message.content

    def read_proofs_data_batch(self, image_payloads: list[dict]) -> list[str]:
        requests_payload: list[dict] = []
        for idx, image_payload in enumerate(image_payloads):
            requests_payload.append(
                {
                    "custom_id": f"receipt-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.primary_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": RECEIPT_PROMPT},
                                    image_payload,
                                ],
                            }
                        ],
                        **DataReader._sampling_kwargs(),
                        **DataReader._completion_token_kwargs(300),
                    },
                }
            )

        return self._run_chat_batch_requests(
            requests_payload, model_name=self.primary_model
        )

    def extract_data_from_statement_text(self, bank_statement_text: str) -> str:
        """Read statement text and extract transaction information."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": STATEMENT_PROMPT + "\n\n" + bank_statement_text,
            },
        ]
        return self._chat_completion_with_fallback(messages, max_tokens=350)

    def extract_data_from_image_texts(self, bank_statement_text: str) -> str:
        """Backward-compatible alias for statement text extraction."""
        return self.extract_data_from_statement_text(bank_statement_text)

    def extract_statement_data_batch(self, statement_texts: list[str]) -> list[str]:
        requests_payload: list[dict] = []
        for idx, statement_text in enumerate(statement_texts):
            requests_payload.append(
                {
                    "custom_id": f"statement-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.primary_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant.",
                            },
                            {
                                "role": "user",
                                "content": STATEMENT_PROMPT + "\n\n" + statement_text,
                            },
                        ],
                        **DataReader._sampling_kwargs(),
                        **DataReader._completion_token_kwargs(350),
                    },
                }
            )

        return self._run_chat_batch_requests(
            requests_payload, model_name=self.primary_model
        )
