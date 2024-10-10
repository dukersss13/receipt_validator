import os
import base64
import numpy as np
import pandas as pd
from time import time
import mimetypes

from openai import OpenAI
from pyhocon import ConfigFactory
from enum import Enum

from src.utils import setup_openai


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
                 config_path: str= "config.conf"):

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
        :return: data in a df format
        """
        if data_type == DataType.TRANSACTIONS:
            data_path = self.transactions_data_path
        elif data_type == DataType.PROOFS:
            data_path = self.proofs_data_path

        payload = DataReader.create_image_payload(data_path)
        start = time()
        data = DataReader.read_data(payload)
        end = time()
        print(f'Time to read {data_type.name}: {round(end - start, 2)}s')

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

        # Loop through all files in the given directory
        for file_name in files:
            # Get the file's MIME type
            mime_type, _ = mimetypes.guess_type(file_name)
            
            # Process only files that are images
            if mime_type and mime_type.startswith('image/'):
                # Read the file and encode it to base64
                image_path = os.path.join(data_path, file_name) if is_dir else file_name
                encoded_string = DataReader.encode_image(image_path)

                # Create the dictionary in the specified format
                image_dict = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_string}",
                    },
                }
                # Append the dictionary to the list
                image_payload.append(image_dict)

        return image_payload

    @staticmethod
    def read_data(image_payload: list[dict]) -> str:
        """
        Read the data to extract information

        :param image_payload: image payload
        :return: response from LLM
        """
        response = client.chat.completions.create(
        model = "gpt-4o-mini",
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

    @staticmethod
    def encode_image(image_path: str):
        """
        Function to encode the image

        :param image_path: path to image (online or offline)
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        