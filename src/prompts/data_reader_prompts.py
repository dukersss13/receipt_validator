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
- Final charged total amount only (numeric, without any currency symbol or code)
- Transaction date (in mm-dd-yyyy format)
- Currency (as an uppercase 3-letter ISO 4217 currency code, e.g., USD, EUR, GBP, JPY)

Rules for amount extraction:
- Use the final paid amount (often labeled as TOTAL, AMOUNT PAID, GRAND TOTAL).
- Do not use subtotal, tax, tip, discount, balance due, or line-item totals.
- Keep cents precision exactly as shown on the receipt.

Rules for output format:
- Return only a Python list of tuples.
- Do not wrap the result in markdown code fences.
- Each tuple must be: (business_name: str, total_amount: float, date: str, currency: str)

Recognize and handle currency in either:
- Symbol form: $, €, £, ¥, ₩, ₹, ₱, etc.
- Code form: USD, EUR, GBP, JPY, KRW, INR, PHP, etc.

If no currency symbol or code is present, assume the currency is USD.

Only return the list. Do not include any explanation or commentary.

Example output:
[('Starbucks', 5.00, '01-15-2023', 'USD'),
 ('Pret A Manger', 7.50, '02-12-2023', 'GBP'),
 ('7-Eleven Japan', 1200.00, '03-05-2023', 'JPY'),
 ('Paris Café', 9.80, '04-18-2023', 'EUR')]
"""
