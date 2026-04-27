# Receipt Validator

## Overview

### Multi-Modal Transaction Validator System🧾
A smart pipeline that extracts, validates, and reconciles financial transactions from statements and proof documents (PDFs or images), using LLMs for intelligent matching and discrepancy detection.

## Architecture 🛠️

The system consists of the following key components:

1.  **Input Layer:**
    * **Transactions:** Statement files (PDFs and supported images) containing transaction records.
    * **Proofs:** Receipt/proof images (and related documents) used to verify transactions.

2.  **Shared LLM Foundation (`src/intelligence/llm_base.py`):**
    * **LLMBase:** A common base class used by LLM-powered components.
    * Loads section-specific model config from `llm_config.conf` (`llm.<section>.*`).
    * Initializes Gemini chat/client objects consistently.
    * Provides shared helpers like API-key resolution and streaming/text extraction utilities.

3.  **Data Ingestion (`src/data/data_reader.py`):**
    * Handles both statement and proof ingestion.
    * **Images:** Encoded to base64 payloads and sent to Gemini for structured extraction.
    * **PDFs:** Read with `PyPDFLoader`, sanitized, then sent to Gemini for extraction.
    * Normalizes extracted rows into a typed schema (`business_name`, `total`, `date`, `currency`).
    * Converts non-USD totals through the currency conversion utility.
    * Tracks ingestion token usage and estimated cost.

4.  **Validation + Intelligence (`src/intelligence/validator.py`, `src/intelligence/categorize.py`, `src/intelligence/helper_agent.py`):
    * **TransactionCategorizer:** Uses Gemini to assign categories to transaction/proof rows.
    * **Validator:** Performs fuzzy business-name matching, date normalization, amount reconciliation, and discrepancy/unmatched analysis.
    * **ArVee HelperAgent:** Tool-calling assistant (`breakdown_spending`) for conversational analysis over validated rows.
    * **Current Agent Scope:** Provides aggregation-based insights only (for example: sum/total spending, average spending per category, and timeframe-based summaries/trends).
    * **In-Chat Memory:** Maintains conversation context within the active chat, so follow-up questions can reference prior turns in the current conversation. ArVee HelperAgent always has context over **validated transactions** at current time, so feel free to ask him anything!

5.  **Web/UI & Persistence (`webui/`, `src/data/database.py`):**
    * Flask web app provides upload, validation, result tables, and chat endpoints.
    * Each session can be saved and loaded via a Session ID (`session_id`) for resume/load workflows.
    * Users can download validated transaction outputs as CSV.

6. **Architecture Diagram**
<img width="1626" height="967" alt="image" src="https://github.com/user-attachments/assets/61532616-8543-4471-ac28-c63a5b4835c7" />



7. **LangGraph Visualization**

   ![image](https://github.com/user-attachments/assets/7e5d07ef-6ffe-4ebd-9bd5-cb6e22c74706)
   



## Set Up ⚙️
To get started, you will need to install:

1. IDE of choice ([VSCode](https://code.visualstudio.com/download) recommended)
2. [Docker](https://www.docker.com/products/docker-desktop/) 
3. Generate a Gemini API key and store it under **secrets/google_gemini_api_key** (or set `GEMINI_API_KEY`).

### RV in Action 💻
Refer to [this](https://github.com/dukersss13/receipt_validator/blob/main/application.md) to see the application in action.

### Custom Website UI
This repository now includes a custom website UI powered by Flask.

1. Install dependencies:
    `pip install -r requirements.txt`
2. Run the web app:
    `python3 webui/app.py`
3. Open your browser at:
    `http://localhost:7860`

The website supports session generation, uploading transaction/proof files, running validation, viewing results tables, and downloading validated CSV records.
Each session can be saved and loaded via a Session ID (`session_id`), and extracted transaction/proof inputs are persisted so previous sessions can be restored in the UI.


 ## 📌 TODO
Extend ArVee HelperAgent Capability
