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
from pathlib import Path

from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from pyhocon import ConfigFactory
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from src.prompts import RECEIPT_PROMPT, STATEMENT_PROMPT
from src.utils.currency_conversion_agent import convert_currency_to_usd
from src.utils.utils import setup_openai, GPT_MODEL
from src.data.database import DataBase


setup_openai()
client = OpenAI()


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
        self.transactions_data_path = data_path["transactions"]
        self.proofs_data_path = data_path["proofs"]
        self.database = database
        # Concurrency knobs for ingestion performance.
        self.io_max_workers = 8
        self.llm_max_workers = 6
        self.fx_max_workers = 8

        if transactions and proofs and len(transactions) and len(proofs):
            self.transactions_data_path = transactions
            self.proofs_data_path = proofs

    def load_data(self, data_type: DataType):
        """
        Read (preprocess) data given data type

        :param data_type: type of data, transactions or proofs
        :return: data in a df format
        """
        if data_type == DataType.TRANSACTIONS:
            processed_data = self.load_transaction_data(self.transactions_data_path)
        elif data_type == DataType.PROOFS:
            processed_data = self.load_proofs_data(self.proofs_data_path)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        # Keep currency for persistence and conversion pipeline.
        return processed_data

    def load_proofs_data(self, data_path: str | list[str]) -> pd.DataFrame:
        """
        Loads proof data from the specified path.
        """
        start = time()
        payload = DataReader.create_image_payload(data_path, self.io_max_workers)
        data = self.batch_read_data(payload)
        end = time()
        print(f"Time to read proofs: {round(end - start, 2)}s")

        data_vec = []
        for item in data:
            parsed = ast.literal_eval(item)
            data_vec.extend(parsed)

        processed_data = DataReader.preprocess_data(data_vec)

        # Convert currency to USD if necessary
        if (processed_data["currency"] != "USD").any():
            non_usd_data = processed_data[processed_data["currency"] != "USD"]
            entries = non_usd_data.to_dict(orient="records")
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
        start = time()
        data = pd.DataFrame([])

        all_files = DataReader.gather_files(data_path)
        pdf_files = [f for f in all_files if Path(f).suffix.lower() == ".pdf"]
        image_files = [
            f for f in all_files if Path(f).suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]

        def process_pdf(pdf_path: str) -> pd.DataFrame:
            try:
                extracted_data = self.extract_data_from_pdf(pdf_path)
                data_vec = ast.literal_eval(extracted_data)
                return DataReader.preprocess_data(data_vec)
            except Exception as e:
                print(f"Warning: Failed to process PDF {pdf_path}: {e}")
                return pd.DataFrame([])

        if pdf_files:
            with ThreadPoolExecutor(
                max_workers=min(self.io_max_workers, len(pdf_files))
            ) as executor:
                pdf_frames = list(executor.map(process_pdf, pdf_files))

            valid_pdf_frames = [frame for frame in pdf_frames if not frame.empty]
            if valid_pdf_frames:
                data = pd.concat([data, *valid_pdf_frames], axis=0)

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
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text = ""
        for doc in docs:
            text += doc.page_content

        filtered_statement_text = DataReader.strip_sensitive_info(text)
        data = self.extract_data_from_image_texts(filtered_statement_text)

        return data

    def batch_read_data(self, image_payloads: list[dict]) -> list[str]:
        """
        Process multiple image payloads in parallel.
        """
        with ThreadPoolExecutor(
            max_workers=min(self.llm_max_workers, max(1, len(image_payloads)))
        ) as executor:
            results = list(executor.map(self.read_proofs_data, image_payloads))

        return results

    @staticmethod
    def preprocess_data(data_vector: np.ndarray) -> pd.DataFrame:
        """
        Preprocess data post ingestion.
        """
        data = pd.DataFrame(
            data_vector, columns=["business_name", "total", "date", "currency"]
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
        response = client.chat.completions.create(
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
        Read statement text and extract transaction information.
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
