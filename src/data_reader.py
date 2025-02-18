from concurrent.futures import ThreadPoolExecutor
import os
import base64
import numpy as np
import pandas as pd
from time import time
import mimetypes
from PIL import Image
import io

from openai import OpenAI
from pyhocon import ConfigFactory
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
            data_path = self.transactions_data_path
        elif data_type == DataType.PROOFS:
            data_path = self.proofs_data_path

        start = time()
        payload = DataReader.create_image_payload(data_path)
        data = DataReader.read_data(payload)
        end = time()
        print(f"Time to read {data_type.name}: {round(end - start, 2)}s")

        data_list = eval(data)
        data_vec = np.asarray(data_list)
        data_df = DataReader.preprocess_data(data_vec)

        return data_df

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
            img = img.resize((int(width * 0.9), int(height * 0.9)), Image.LANCZOS)
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

    @staticmethod
    def read_data(image_payload: list[dict]) -> str:
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
                "text": """Tell me the business names, totals and transaction dates. 
                        Answer should be formatted like: [('name1', 'total1', 'date1'), ('name2', 'total2', 'date2'), etc...].
                        The business names should be strings.
                        The dates should be formatted as mm-dd-yyyy.
                        Give the total amount as numeric. No need to include the currency denomination.
                        Only give me the list, nothing else.""",
                },
                *image_payload
            ],
            }
        ],
        max_tokens=1000,
        )

        return response.choices[0].message.content
        