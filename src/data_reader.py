import os
import base64
import numpy as np
import pandas as pd
from time import time
import mimetypes
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import io

from openai import OpenAI
from pyhocon import ConfigFactory
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from src.utils import setup_openai, GPT_MODEL


setup_openai()
client = OpenAI()


class DataType(Enum):
    TRANSACTIONS = 1
    PROOFS = 2


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
            data = self.load_data(self.transactions_data_path)
        elif data_type == DataType.PROOFS:
            data = self.load_data(self.proofs_data_path)

        data_list = [eval(line)[0] for line in data]
        data_vec = np.asarray(data_list)
        data_df = DataReader.preprocess_data(data_vec)

        return data_df

    def load_images_with_openai(self, data_path: str) -> str:
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

        return data

    def load_data(self, data_path: str) -> str:
        """
        Loads transaction data from a given file path.
        This method reads images from the specified file path, extracts text from the images using OCR,
        and processes the extracted text to read transaction statements.
        Args:
            data_path (str): The file path to the transaction data.
        Returns:
            str: The processed transaction data.
        Prints:
            The time taken to read and process the transaction statements.
        """
        start = time()
        images = convert_from_path(data_path)
        full_statement_text = DataReader.convert_image_to_text(images)
        
        data = self.extract_data_from_image_texts(full_statement_text)
        end = time()
        print(f"Time to read transaction statements: {round(end - start, 2)}s")

        return data
    
    @staticmethod
    def convert_image_to_text(images) -> list[str]:
        """_summary_

        Args:
            image: _description_

        Returns:
            _description_
        """
        full_text = ""
        for img in images:
            text = pytesseract.image_to_string(img)
            full_text += text

        return text

    def batch_read_data(self, text_payloads: list[list[dict]]) -> str:
        """
        Process multiple image payloads in parallel
        """
        with ThreadPoolExecutor() as executor:  # Adjust based on API rate limits
            results = list(executor.map(self.extract_data_from_image_texts, text_payloads))

        return results

    @staticmethod
    def preprocess_data(data_vector: np.ndarray) -> pd.DataFrame:
        """
        Preprocess data post ingestion
        """
        data = pd.DataFrame(data_vector, columns=["business_name", "total", "date"])
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

    @property
    def prompt(self):
        # Prompt
        client_prompt = \
        """
        Tell me the business names, totals and transaction dates. 
        Answer should be formatted like: [('name1', 'total1', 'date1'), ('name2', 'total2', 'date2'), etc...].
        The business names should be strings.
        The dates should be formatted as mm-dd-yyyy.
        Give the total amount as numeric. No need to include the currency denomination.
        Only give me the list, nothing else.
        """
        return client_prompt

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
                "text": self.prompt,
                },
                image_payload
            ],
            }
        ],
        max_tokens=200,
        )

        return response.choices[0].message.content

    def extract_data_from_image_texts(self, bank_statement_text: str) -> str:
        """
        Read the data to extract information

        :param image_payload: image payload
        :return: response from LLM
        """
        response = client.chat.completions.create(
            model=GPT_MODEL,  # Use "gpt-4-turbo" or "gpt-4" depending on your access
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.prompt + "\n\n" + bank_statement_text}
            ],
            temperature=0,
            max_tokens=200
        )

        return response.choices[0].message.content
