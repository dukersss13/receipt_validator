import os
import base64
import numpy as np
import pandas as pd
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
    def __init__(self, config_path: str= "config.conf"):
        data_path = ConfigFactory.parse_file(config_path).get("data_path")
        self.transactions_data_path = data_path["transactions"]
        self.proofs_data_path = data_path["proofs"]

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
        data = DataReader.read_data(payload)
        data_list = eval(data)
        data_vec = np.asarray(data_list)

        data_df = pd.DataFrame(data_vec, columns=["business_name", "total", "date"])
        data_df["total"] = data_df["total"].astype(float)

        return data_df

    @staticmethod
    def create_image_payload(data_path: str) -> list[dict]:
        """
        Create image payload

        :param data_path: data path
        :return: payload for encoded images
        """
        image_payload = []

        # Loop through all files in the given directory
        for file_name in os.listdir(data_path):
            # Get the file's MIME type
            mime_type, _ = mimetypes.guess_type(file_name)
            
            # Process only files that are images
            if mime_type and mime_type.startswith('image/'):
                # Read the file and encode it to base64
                image_path = os.path.join(data_path, file_name)
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
                        Answer should be formatted like: [(name1, total1, date1), (name2, total2, date2), etc...].
                        Only give me the list, nothing else.""",
                },
                *image_payload
            ],
            }
        ],
        max_tokens=300,
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
        