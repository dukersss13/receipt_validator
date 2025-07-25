# Receipt Validator

## Overview

### Multi-Modal Transaction Validator System🧾
A smart pipeline that extracts, validates, and reconciles financial transactions from statements and proof documents (PDFs or images), using LLMs for intelligent matching and discrepancy detection.

## Architecture 🛠️

The system consists of the following key components:

1.  **Input:**
    * **Transactions:** A collection of transaction records, typically bank statement PDFs or images.
    * **Proofs:** Supporting documents for the transactions, such as receipts, typically images.

2.  **Data Reader:**
    * **Transaction Statements & Proofs🗂️:** This component receives the input transactions and proofs.
    * **Images:** Proof receipt images are decoded into base64 strings, which then get sent to ChatGPT API for data extraction.
    * **PDFs:** Handles PDF-based proofs.
        * **PyPDF Loader:** A module responsible for reading and extracting text content from PDF files.
        * **PDF Text:** The extracted text from the PDF proofs.
    * **Image Payload & PDF Text:** Processed image payloads and/or PDF texts from statements get sent to ChatGPT API for data extraction (business names, transaction totals & dates).

3.  **Validator:**
    * **Discrepancy Calculator:** This module compares the transaction data against the information extracted from the proofs. It identifies discrepancies in amounts, dates, or other relevant fields. An example in the diagram shows a "Lego - 2/11/25 - $2 discrepancy."
    * **Transactions Matching:** This module attempts to match individual transactions with corresponding entries in the proofs. Matching is done through fuzzymatching with threshold of 80%.
    * **Recommendation Agent🧠:** This module takes the "Unmatched Transactions" and uses the OpenAI API to potentially generate recommendations (e.g., further investigation, manual review).
    * **Output:**
        * **Validated Transactions✅:** Transactions that have been successfully matched and validated against the proofs.
        * **Discrepancy Information⚠️:** Details of any discrepancies found (e.g., "Lego - 2/11/25 - $2 discrepancy").
        * **Unmatched Transactions❓:** Transactions that could not be matched with any information in the provided proofs.

4.  **OpenAI API:**
    * The OpenAI API is utilized by the "Recommendation Agent." The diagram shows the API receiving "Business Names, Totals and Dates" and "Unmatched Transactions" as input and providing "Recommendations" as output. It is also used by the Data Reader for data extraction.

5.  **Output (Overall System):**
    * **Validated Transactions:** A structured output of transactions that have been successfully verified.
    * **Discrepancy:** A record of all identified discrepancies.
    * **Recommendations:** Suggestions generated by the OpenAI API for handling unmatched transactions.
  Users are allowed to download the validated records of the transactions-proofs output.

6. **Architecture Diagram**
![Screenshot 2025-05-15 at 1 31 12 PM](https://github.com/user-attachments/assets/9ee7244c-b785-4d6f-8388-bc1cc4e2be3a)

7. **LangGraph Visualization**

   ![image](https://github.com/user-attachments/assets/7e5d07ef-6ffe-4ebd-9bd5-cb6e22c74706)
   



## Set Up ⚙️
To get started, you will need to install:

1. IDE of choice ([VSCode](https://code.visualstudio.com/download) recommended)
2. [Docker](https://www.docker.com/products/docker-desktop/) 
3. Generate [OpenAI API Key](https://openai.com/index/openai-api/) & store under **secrets/openai_api_key**

### RV in Action 💻
Refer to [this](https://github.com/dukersss13/receipt_validator/blob/main/application.md) to see the application in action.


 ## 📌 TODO
 Improve image OCR robustness

 Support more languages/currencies
