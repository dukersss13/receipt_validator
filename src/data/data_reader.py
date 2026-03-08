import os
import asyncio
import base64
import numpy as np
import pandas as pd
from time import time
import mimetypes
from PIL import Image
import io
import re
import ast
from pathlib import Path

from openai import OpenAI, AsyncOpenAI
from langchain_community.document_loaders import PyPDFLoader
from pyhocon import ConfigFactory
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

try:
    import pytesseract
except Exception:
    pytesseract = None

from src.prompts import RECEIPT_PROMPT, STATEMENT_PROMPT
from src.utils.currency_conversion_agent import convert_entries_to_usd_async
from src.utils.utils import setup_openai, GPT_MODEL
from src.data.database import DataBase


setup_openai()
client = OpenAI()
async_client = AsyncOpenAI()


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
    ):

        data_path = ConfigFactory.parse_file(config_path).get("data_path")
        self.database = database or DataBase(
            engine_name="receipt_validator_db", local_db=True
        )
        self.transactions_data_path = data_path["transactions"]
        self.proofs_data_path = data_path["proofs"]

        if transactions and proofs and len(transactions) and len(proofs):
            self.transactions_data_path = transactions
            self.proofs_data_path = proofs

        # Tune for throughput vs. provider rate limits.
        self.api_max_workers = 6
        self.receipt_batch_size = 4
        self.max_receipt_chars = 2800

    def load_files(
        self, transactions: list[str] | None = None, proofs: list[str] | None = None
    ):
        """
        Load files into the DataReader instance.
        """
        self.transactions = transactions
        self.proofs = proofs

    def load_data(self, data_type: DataType):
        """
        Read (preprocess) data given data type

        :param data_type: type of data, transactions or proofs
        :return: data in a df formatz
        """
        if data_type == DataType.TRANSACTIONS:
            processed_data = self.load_transaction_data(self.transactions_data_path)
        elif data_type == DataType.PROOFS:
            processed_data = self.load_proofs_data(self.proofs_data_path)

        return processed_data  # Drop currency column for validation

    def load_proofs_data(self, data_path: str | list[str]) -> pd.DataFrame:
        """
        Loads proof data from the specified path.
        Args:
            data_path (str): The path to the data file.
        Returns:
            str: The loaded data as a string.
        """
        start = time()
        all_files = self.gather_files(data_path)
        image_files = [
            file_path
            for file_path in all_files
            if Path(file_path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
        ]

        if not image_files:
            return DataReader.preprocess_data([])

        # Step 1: OCR locally to avoid expensive image-token usage.
        with ThreadPoolExecutor(max_workers=min(8, len(image_files))) as executor:
            ocr_texts = list(executor.map(self.extract_receipt_text, image_files))

        # Step 2: Send OCR text in batches for structured extraction.
        text_chunks = []
        vision_fallback_paths = []
        for file_path, text in zip(image_files, ocr_texts):
            cleaned_text = (text or "").strip()
            if cleaned_text:
                text_chunks.append(cleaned_text[: self.max_receipt_chars])
            else:
                vision_fallback_paths.append(file_path)

        text_batches = [
            text_chunks[i : i + self.receipt_batch_size]
            for i in range(0, len(text_chunks), self.receipt_batch_size)
        ]

        data = []
        if text_batches:
            data.extend(self.batch_read_data(text_batches))

        # Step 3: Fallback to vision only when OCR fails.
        if vision_fallback_paths:
            fallback_payload = DataReader.create_image_payload(vision_fallback_paths)
            data.extend(self.batch_read_vision_data(fallback_payload))

        end = time()
        print(f"Time to read proofs: {round(end - start, 2)}s")

        data_vec = []
        for item in data:
            try:
                parsed = ast.literal_eval(item)
                data_vec.extend(parsed)
            except Exception:
                print("Warning: Failed to parse extracted proof payload; skipping a batch.")

        processed_data = DataReader.preprocess_data(data_vec)

        # Convert currency to USD if necessary
        if (processed_data["currency"] != "USD").any():
            non_usd_data = processed_data[processed_data["currency"] != "USD"]
            conversion_entries = non_usd_data.to_dict(orient="records")
            converted_totals = asyncio.run(
                convert_entries_to_usd_async(
                    conversion_entries,
                    max_concurrency=self.api_max_workers,
                )
            )
            processed_data.loc[non_usd_data.index, "total"] = converted_totals

        return processed_data

    @staticmethod
    def strip_sensitive_info(text):
        # Remove email addresses
        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", text)

        # Remove phone numbers (basic US formats)
        text = re.sub(
            r"\b(?:\+?1[-.\s]?)*\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text
        )

        # Remove dates (MM/DD/YYYY, DD-MM-YYYY, etc.)
        text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE]", text)

        # Remove ZIP codes
        text = re.sub(r"\b\d{5}(?:-\d{4})?\b", "[ZIP]", text)

        # Remove credit card numbers (13 to 16 digits)
        text = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[CREDIT_CARD]", text)

        # Remove account numbers (common formats: 8–12 digits, with or without label)
        text = re.sub(
            r"\b(?:Account|Acct|A/C)[\s#:]*\d{8,20}\b",
            "[ACCOUNT]",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\b\d{8,20}\b", "[ACCOUNT]", text)  # bare numbers fallback

        # Remove street addresses (basic version)
        text = re.sub(
            r"\d{1,5}\s+\w+(\s+\w+)*\s+(Street|St|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane)\b",
            "[ADDRESS]",
            text,
            flags=re.IGNORECASE,
        )

        # Remove full names in Title Case (e.g., John Smith)
        text = re.sub(r"\b([A-Z][a-z]+\s[A-Z][a-z]+)\b", "[NAME]", text)

        return text

    def load_transaction_data(self, data_path: str | list[str]) -> pd.DataFrame:
        """
        Loads transaction data from PDFs and image files in a given path or list of file paths.
        Applies OCR and LLM extraction to return a structured DataFrame.

        Args:
            data_path (str | list[str]): Path to a directory, single file, or list of files.

        Returns:
            pd.DataFrame: Structured transaction data with business name, total, and date.
        """
        start = time()
        data = pd.DataFrame([])

        # Collect and separate files
        all_files = self.gather_files(data_path)
        pdf_files = [f for f in all_files if Path(f).suffix.lower() == ".pdf"]
        image_files = [
            f for f in all_files if Path(f).suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]

        # Process PDFs
        for pdf_path in pdf_files:
            try:
                extracted_data = self.extract_data_from_pdf(pdf_path)
                data_vec = ast.literal_eval(extracted_data)
                processed_data = DataReader.preprocess_data(data_vec)
                data = pd.concat([data, processed_data], axis=0)
            except Exception as e:
                print(f"Warning: Failed to process PDF {pdf_path}: {e}")

        # Process images (batch)
        if image_files:
            try:
                image_data = self.load_proofs_data(image_files)
                data = pd.concat([data, image_data], axis=0)
            except Exception as e:
                print(f"Warning: Failed to process image files: {e}")

        end = time()
        print(f"Time to read transaction statements: {round(end - start, 2)}s")

        return data

    @staticmethod
    def gather_files(data_path: str | list[str]) -> list[str]:
        """Resolve an input path/list into a concrete list of files."""
        if isinstance(data_path, str):
            path_obj = Path(data_path)
            if path_obj.is_file():
                return [str(path_obj)]
            if path_obj.is_dir():
                return [str(file_path) for file_path in path_obj.iterdir() if file_path.is_file()]
        elif isinstance(data_path, list):
            return [str(file_path) for file_path in data_path]

        raise ValueError(
            "data_path must be a file path, directory path, or list of file paths."
        )

    def extract_data_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts data from a PDF file located at the specified path.
        Args:
            data_path (str): The path to the PDF file.
        Returns:
            str: The extracted data as a string.
        """
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text = ""
        for doc in docs:
            text += doc.page_content

        filtered_statement_text = DataReader.strip_sensitive_info(text)
        data = self.extract_data_from_image_texts(filtered_statement_text)

        return data

    def batch_read_data(self, receipt_text_batches: list[list[str]]) -> list[str]:
        """
        Process multiple OCR text batches in parallel.
        """
        if not receipt_text_batches:
            return []

        return asyncio.run(self._run_async_receipt_text_batches(receipt_text_batches))

    def batch_read_vision_data(self, image_payloads: list[dict]) -> list[str]:
        """Fallback path: process raw image payloads in parallel."""
        if not image_payloads:
            return []

        return asyncio.run(self._run_async_receipt_image_batches(image_payloads))

    async def _run_async_receipt_text_batches(
        self, receipt_text_batches: list[list[str]]
    ) -> list[str]:
        """Execute text extraction requests concurrently with bounded fan-out."""
        semaphore = asyncio.Semaphore(max(1, self.api_max_workers))

        async def run_with_limit(batch: list[str]) -> str:
            async with semaphore:
                return await self.read_proofs_data_async(batch)

        tasks = [run_with_limit(batch) for batch in receipt_text_batches]
        return await asyncio.gather(*tasks)

    async def _run_async_receipt_image_batches(self, image_payloads: list[dict]) -> list[str]:
        """Execute vision fallback requests concurrently with bounded fan-out."""
        semaphore = asyncio.Semaphore(max(1, self.api_max_workers))

        async def run_with_limit(payload: dict) -> str:
            async with semaphore:
                return await self.read_proofs_data_from_image_async(payload)

        tasks = [run_with_limit(payload) for payload in image_payloads]
        return await asyncio.gather(*tasks)

        return results

    @staticmethod
    def preprocess_data(data_vector: np.ndarray) -> pd.DataFrame:
        """
        Preprocess data post ingestion
        """
        data = pd.DataFrame(
            data_vector, columns=["business_name", "total", "date", "currency"]
        )
        data["business_name"] = data["business_name"].str.lower()
        data["total"] = data["total"].astype(float)

        return data

    @staticmethod
    def create_image_payload(data_path: str | list[str]) -> list[dict]:
        """
        Create image payload

        :param data_path: data path
        :return: payload for encoded images
        """
        image_payload = []
        is_dir = isinstance(data_path, str)
        files = os.listdir(data_path) if is_dir else data_path

        def process_file(file_name):
            # Get the file's MIME type
            mime_type, _ = mimetypes.guess_type(file_name)

            # Process only files that are images
            if mime_type and mime_type.startswith("image/"):
                # Read the file and encode it to base64
                image_path = os.path.join(data_path, file_name) if is_dir else file_name
                encoded_string = DataReader.encode_image(image_path)

                # Create the dictionary in the specified format
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_string}",
                    },
                }

            return None

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_file, files))

        # Filter out None results
        image_payload = [result for result in results if result is not None]

        return image_payload

    @staticmethod
    def reduce_image_size(image_path, max_size=2 * 1024 * 1024):
        """
        Reduce an image's size to be under max_size (default 2MB) without saving to disk.

        :param image_path: Path to the input image
        :param max_size: Maximum allowed file size in bytes (default 2MB)
        :return: Compressed image as an in-memory bytes object
        """
        img = Image.open(image_path)

        # Convert to RGB (handles transparency in PNGs)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Initialize a BytesIO object to store the compressed image
        img_bytes = io.BytesIO()

        # Start with a high quality and progressively reduce
        quality = 90
        while True:
            img_bytes.seek(0)  # Reset buffer position
            img_bytes.truncate(0)  # Clear buffer
            img.save(img_bytes, format="JPEG", quality=quality)

            # Check if the file size is within the limit
            if img_bytes.tell() <= max_size or quality <= 10:
                break  # Stop if it's under max_size or quality is too low

            # Reduce quality
            quality -= 5

        # If still too large, resize proportionally
        while img_bytes.tell() > max_size:
            width, height = img.size
            img = img.resize((int(width * 0.95), int(height * 0.95)), Image.LANCZOS)
            img_bytes.seek(0)
            img_bytes.truncate(0)
            img.save(img_bytes, format="JPEG", quality=quality)

        img_bytes.seek(0)  # Reset buffer to the start for reading

        return img_bytes

    @staticmethod
    def encode_image(image_path: str):
        """
        Function to encode the image

        :param image_path: path to image (online or offline)
        """
        img_bytes = DataReader.reduce_image_size(image_path)
        return base64.b64encode(img_bytes.read()).decode("utf-8")

    def extract_receipt_text(self, image_path: str) -> str:
        """Run local OCR on receipt image to reduce image-token usage."""
        if pytesseract is None:
            return ""

        try:
            image = Image.open(image_path)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")

            text = pytesseract.image_to_string(image, config="--psm 6")
            return text.strip()
        except Exception:
            return ""

    async def read_proofs_data_async(self, receipt_text_batch: list[str]) -> str:
        """
        Extract receipt structured fields from OCR text batch.

        :param receipt_text_batch: list of OCR text blocks
        :return: response from LLM
        """
        batch_text = "\n\n---\n\n".join(receipt_text_batch)

        response = await async_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        RECEIPT_PROMPT
                        + "\n\nThe following are OCR texts from one or more receipts. "
                        + "Extract all transactions found and return only one flattened list of tuples.\n\n"
                        + batch_text
                    ),
                }
            ],
            temperature=0,
            max_tokens=700,
        )

        return response.choices[0].message.content

    async def read_proofs_data_from_image_async(self, image_payload: dict) -> str:
        """Vision fallback path for a single receipt image payload."""
        response = await async_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
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
            ],
            temperature=0,
            max_tokens=300,
        )

        return response.choices[0].message.content

    def extract_data_from_image_texts(self, bank_statement_text: str) -> str:
        """
        Read the data to extract information

        :param image_payload: image payload
        :return: response from LLM
        """
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": STATEMENT_PROMPT + "\n\n" + bank_statement_text,
                },
            ],
            temperature=0,
            max_tokens=350,
        )

        return response.choices[0].message.content
