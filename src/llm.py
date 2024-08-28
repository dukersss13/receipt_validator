from openai import OpenAI
from utils import setup_openai


setup_openai()
client = OpenAI()


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
