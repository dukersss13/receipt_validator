import os
import mimetypes

from openai import OpenAI
from helper.helper_functions import encode_image
from helper.utils import setup_openai


setup_openai()
client = OpenAI()


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
            encoded_string = encode_image(image_path)

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


def analyze_receipts(image_payload: list[dict]) -> str:
    """
    Analyze the receipts to extract the business names
    and totals

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
            "text": """Tell me the business names and totals. 
                    Answer should be formatted like: [(name1, total1), (name2, total2), etc...].
                    Only give me the list, nothing else.""",
            },
            *image_payload
        ],
        }
    ],
    max_tokens=300,
    )

    return response.choices[0].message.content

image_payload = create_image_payload(data_path="data/receipts")

receipts_info = analyze_receipts(image_payload)

x = 3
