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

from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from pyhocon import ConfigFactory
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from src.currency_conversion_agent import convert_currency_to_usd
from src.utils import setup_openai, GPT_MODEL


setup_openai()
client = OpenAI()


class DataType(Enum):
    TRANSACTIONS = "transactions"
    PROOFS = "proofs"


STATEMENT_PROMPT = """
You are given a block of text from a bank statement.
The text contains information about transactions, including business names, totals, and transaction dates.
Be sure to only extract the purchases and ignore any other information such as payments made.
Payments are usually noted with a negative sign, such as -$5.00.
Your task is to extract this information and format it as a list of tuples.
Each tuple should contain the business name, total amount, and transaction date.
The business names should be strings.
The dates should be formatted as mm-dd-yyyy.
The total amount should be numeric, without any currency denomination.
Only give me the list, nothing else.

For example, if the text contains:
"Transaction at Starbucks on 01-15-2023 for $5.00"
You should return:
[('Starbucks', 5.00, '01-15-2023')] 

If there are multiple transactions, separate them with commas.

For example:
"Transaction at Starbucks on 01-15-2023 for $5.00, Transaction at Amazon on 01-16-2023 for $20.00"
You should return:
[('Starbucks', 5.00, '01-15-2023'), ('Amazon', 20.00, '01-16-2023')]

Make sure to format the output correctly.
Do not include any additional text or explanations.
"""

RECEIPT_PROMPT = """
You are given a block of text from a receipt image.

The image contains information about transactions,
including business names, totals, transaction dates, and possibly foreign currency symbols or codes.

Your task is to extract the following for each transaction:
- Business name (string)
- Total amount (numeric, without any currency symbol or code)
- Transaction date (in mm-dd-yyyy format)
- Currency (as an uppercase 3-letter ISO 4217 currency code, e.g., USD, EUR, GBP, JPY)

Recognize and handle currency in either:
- Symbol form: $, €, £, ¥, ₩, ₹, ₱, etc.
- Code form: USD, EUR, GBP, JPY, KRW, INR, PHP, etc.

If no currency symbol or code is present, assume the currency is USD.

Format your output as a list of tuples:
(business_name: str, total_amount: float, date: str, currency: str)

Only return the list. Do not include any explanation or commentary.

Example output:
[('Starbucks', 5.00, '01-15-2023', 'USD'),
 ('Pret A Manger', 7.50, '02-12-2023', 'GBP'),
 ('7-Eleven Japan', 1200.00, '03-05-2023', 'JPY'),
 ('Paris Café', 9.80, '04-18-2023', 'EUR')]
"""



class DataReader:
    """
    Data Reader class ingests transactions and proofs images
    prior to the validation process.
    """
    def __init__(self, transactions: list[str] | None = [], 
                 proofs: list[str] | None = [],
                 config_path: str = "config.conf"):

        data_path = ConfigFactory.parse_file(config_path).get("data_path")
        self.transactions_data_path = data_path["transactions"]
        self.proofs_data_path = data_path["proofs"]

        if transactions and proofs and len(transactions) and len(proofs):
            self.transactions_data_path = transactions
            self.proofs_data_path = proofs

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

        return processed_data

    def load_proofs_data(self, data_path: str | list[str]) -> pd.DataFrame:
        """
        Loads proof data from the specified path.
        Args:
            data_path (str): The path to the data file.
        Returns:
            str: The loaded data as a string.
        """
        start = time()
        payload = DataReader.create_image_payload(data_path)
        data = self.batch_read_data(payload)
        end = time()
        print(f"Time to read proofs: {round(end - start, 2)}s")

        # Flattened list of tuples
        data_vec = []

        for item in data:
            parsed = ast.literal_eval(item)  # safely parse string into Python object
            data_vec.extend(parsed) # Flatten the list of tuples 

        processed_data = DataReader.preprocess_data(data_vec)

        # Convert currency to USD if necessary
        if (processed_data["currency"] != "USD").any():
            non_usd_data = processed_data[processed_data["currency"] != "USD"]
            processed_data.loc[non_usd_data.index, "total"] = non_usd_data.apply(lambda row: \
                                                                                 convert_currency_to_usd(row.to_dict()), axis=1)
        
        processed_data["currency"] = "USD"  # Set all currencies to USD after conversion
    
        return processed_data
    
    @staticmethod
    def strip_sensitive_info(text):
        # Remove email addresses
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)

        # Remove phone numbers (basic US formats)
        text = re.sub(r'\b(?:\+?1[-.\s]?)*\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)

        # Remove dates (MM/DD/YYYY, DD-MM-YYYY, etc.)
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)

        # Remove ZIP codes
        text = re.sub(r'\b\d{5}(?:-\d{4})?\b', '[ZIP]', text)

        # Remove credit card numbers (13 to 16 digits)
        text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[CREDIT_CARD]', text)

        # Remove account numbers (common formats: 8–12 digits, with or without label)
        text = re.sub(r'\b(?:Account|Acct|A/C)[\s#:]*\d{8,20}\b', '[ACCOUNT]', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d{8,20}\b', '[ACCOUNT]', text)  # bare numbers fallback

        # Remove street addresses (basic version)
        text = re.sub(r'\d{1,5}\s+\w+(\s+\w+)*\s+(Street|St|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane)\b', 
                      '[ADDRESS]', text, flags=re.IGNORECASE)

        # Remove full names in Title Case (e.g., John Smith)
        text = re.sub(r'\b([A-Z][a-z]+\s[A-Z][a-z]+)\b', '[NAME]', text)

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
        from pathlib import Path

        start = time()
        data = pd.DataFrame([])

        # Resolve files from the input
        def gather_files(path_input):
            if isinstance(path_input, str):
                p = Path(path_input)
                if p.is_file():
                    return [str(p)]
                elif p.is_dir():
                    return [str(f) for f in p.iterdir() if f.is_file()]
            elif isinstance(path_input, list):
                return path_input
            raise ValueError("data_path must be a file path, directory path, or list of file paths.")

        # Collect and separate files
        all_files = gather_files(data_path)
        pdf_files = [f for f in all_files if Path(f).suffix.lower() == ".pdf"]
        image_files = [f for f in all_files if Path(f).suffix.lower() in {".png", ".jpg", ".jpeg"}]

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

    def batch_read_data(self, image_payload: list[list[dict]]) -> str:
        """
        Process multiple image payloads in parallel
        """
        with ThreadPoolExecutor() as executor:  # Adjust based on API rate limits
            results = list(executor.map(self.read_proofs_data, image_payload))

        return results

    @staticmethod
    def preprocess_data(data_vector: np.ndarray) -> pd.DataFrame:
        """
        Preprocess data post ingestion
        """
        data = pd.DataFrame(data_vector, columns=["business_name", "total", "date", "currency"])
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
    def reduce_image_size(image_path, max_size=2*1024*1024):
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
        return base64.b64encode(img_bytes.read()).decode('utf-8')

    def read_proofs_data(self, image_payload: list[dict]) -> str:
        """
        Read the data to extract information

        :param image_payload: image payload
        :return: response from LLM
        """
        response = client.chat.completions.create(
        model = GPT_MODEL,
        messages = [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": RECEIPT_PROMPT,
                },
                image_payload
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
                {"role": "user", "content": STATEMENT_PROMPT + "\n\n" + bank_statement_text}
            ],
            temperature=0,
            max_tokens=350
        )

        return response.choices[0].message.content
