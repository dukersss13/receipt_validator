import pytest
import os

from openai import OpenAI
import pandas as pd
from pdf2image import convert_from_path
import pytesseract
from PIL import Image


from src.utils import GPT_MODEL, setup_openai

setup_openai()
client = OpenAI()

pd.set_option("display.max_columns", None)


chase_statement_path = "data/transactions/chase_statement.pdf"
sofi_statement_path = "data/transactions/sofi_statement.pdf"
statements_path = "data/transactions/"
proofs_path = "data/proofs/"


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping on GitHub Actions"
)
def test_statement_reading():
    # List all files in the directory
    statements = [f for f in os.listdir(statements_path) if f.lower().endswith((".pdf"))]

    texts = []

    # Process each image
    for statement in statements:
        image_path = os.path.join(statements_path, statement)
        images = convert_from_path(image_path, dpi=300)

        # Convert image to text using pytesseract
        for img in images:
            extracted_text = pytesseract.image_to_string(img, config="--psm 6")
            texts.append(extracted_text)
        
        texts = " ".join(texts)
        break

    # The system message instructs the model on how to respond
    prompt = """
    You are given a block of text from a bank statement or a receipt image.
    The text contains information about transactions, including business names, totals, and transaction dates.
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

    results = []
    for text in texts:
        # Make the API request
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt + "\n\n" + text}
            ],
            temperature=0,
            max_tokens=500
        )
    
        results.append(response.choices[0].message.content)

    print(results)
    

@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping on GitHub Actions"
)
def test_image_to_string_conversion():
    
    # List all files in the directory
    image_files = [f for f in os.listdir(proofs_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    texts = []

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(proofs_path, image_file)
        # Read the image using PIL
        img = Image.open(image_path)
        
        # Convert image to text using pytesseract
        extracted_text = pytesseract.image_to_string(img, config='--psm 6')
        
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
    