from openai import OpenAI
import pandas as pd
from pdf2image import convert_from_path
import pytesseract
import os
from PIL import Image


from src.utils import GPT_MODEL, setup_openai

setup_openai()
client = OpenAI()

pd.set_option("display.max_columns", None)


chase_statement_path = "data/transactions/chase_statement.pdf"
sofi_statement_path = "data/transactions/sofi_statement.pdf"
proofs_path = "data/proofs/"

import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)  # Apply thresholding
    return Image.fromarray(img)  # Convert back to PIL Image


def test_statement_reading():
    
    # List all files in the directory
    image_files = [f for f in os.listdir(proofs_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    texts = []

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(proofs_path, image_file)
        img = preprocess_image(image_path)
        
        # Convert image to text using pytesseract
        extracted_text = pytesseract.image_to_string(img)
        
        texts.append(extracted_text)

    # The system message instructs the model on how to respond
    prompt = """
    Tell me the business names, totals and transaction dates. 
    Answer should be formatted like: [('name1', 'total1', 'date1'), ('name2', 'total2', 'date2'), etc...].
    The business names should be strings.
    The dates should be formatted as mm-dd-yyyy.
    Give the total amount as numeric. No need to include the currency denomination.
    Only give me the list, nothing else.
    """

    results = []
    for text in texts:
        # Make the API request
        response = client.chat.completions.create(
            model=GPT_MODEL,  # Use "gpt-4-turbo" or "gpt-4" depending on your access
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt + "\n\n" + text}
            ],
            temperature=0,
            max_tokens=200
        )
    
        results.append(response.choices[0].message.content)

    print(results)
    