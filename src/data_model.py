import numpy as np
import pandas as pd
import os
import mimetypes

from helper_functions import encode_image
from llm import analyze_receipts


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


image_payload = create_image_payload(data_path="data/receipts")
receipts_info = analyze_receipts(image_payload)
receipts_list = eval(receipts_info)
receipts_vec = np.asarray(receipts_list)

df_receipts = pd.DataFrame(receipts_vec, columns=["business_name", "total", "date"])
df_receipts["business_name"] = df_receipts["business_name"].astype(str)
df_receipts["total"] = df_receipts["total"].astype(float)

bank_statement = pd.DataFrame({"business_name": ["Taco Bell", "Cider Cellar", "Mogu mogu - Costa Mesa"],
                               "total": [24.39, 4.50, 57.0],
                               "date": receipts_vec[:, -1].astype(str)})

