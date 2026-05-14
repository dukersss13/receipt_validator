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
    * Loads section-specific model config from `config/llm_config.conf` (`llm.<section>.*`).
    * Initializes Gemini chat/client objects consistently.
    * Provides shared helpers like API-key resolution and streaming/text extraction utilities.

3.  **Data Ingestion (`src/data/data_reader.py`):**
    * Handles both statement and proof ingestion.
    * **Images:** Encoded to base64 payloads and sent to Gemini for structured extraction.
    * **PDFs:** Read with `PyPDFLoader`, sanitized, then sent to Gemini for extraction.
    * Normalizes extracted rows into a typed schema (`business_name`, `total`, `date`, `currency`).
    * Converts non-USD totals through the currency conversion utility.
    * Tracks ingestion token usage and estimated cost.

4.  **Validation + Intelligence** (`src/agents/validator.py`, `src/agents/categorize.py`, `src/agents/router_agent.py`, `src/agents/agent_tools.py`):
    * **TransactionCategorizer:** Uses Gemini to assign categories to transaction/proof rows.
    * **Validator:** Performs fuzzy business-name matching, date normalization, amount reconciliation, and discrepancy/unmatched analysis.
    * **ArVee Agent Flow:** `RouterAgent` interprets user intent, extracts structured params, and dispatches directly to `AgentTools` for deterministic execution — no LLM computes numbers.
    * **Current Tools:** `spending_breakdown` and `compare_spending_periods`.
    * **Clarification Flow:** When a query is ambiguous, RouterAgent stores a pending plan, returns a clarification question with quick-reply suggestions, and merges the user's answer with preserved params on the next turn.
    * **Conversation Context:** RouterAgent tracks the last 10 turns internally, so follow-up questions reference prior context.
    * **More Info:** See [ArVee Agent details](md/arvee_agent.md).

5.  **Web/UI & Persistence (`webui/`, `src/data/database.py`):**
    * Flask web app provides upload, validation, result tables, and chat endpoints.
    * Each session can be saved and loaded via a Session ID (`session_id`) for resume/load workflows.
    * Users can download validated transaction outputs as CSV.

6. **Architecture Diagram**

The web UI system architecture diagram now lives in the separate frontend repository:
[dukersss13/arvee_web_ui/docs/system_architecture.svg](https://github.com/dukersss13/arvee_web_ui/blob/main/docs/system_architecture.svg)



## Set Up ⚙️
To get started, you will need to install:

1. IDE of choice ([VSCode](https://code.visualstudio.com/download) recommended)
2. [Docker](https://www.docker.com/products/docker-desktop/) 
3. Generate a Gemini API key and set `GEMINI_API_KEY` via environment.

### Local Environment Configuration

Use a local `.env` file for development only:

1. Create local env file:
    `cp .env.example .env`
2. Fill in required values (at minimum `GEMINI_API_KEY`).
3. Load variables in your shell before running the app:
    `set -a; source .env; set +a`

Do not commit `.env` or any real secret files.

### Production Secrets (GCP)

For Cloud Run deployments, store secrets in Secret Manager and inject them at deploy time.
Do not mount or commit secret files in production images.

### ArVee in Action 💻
Web UI walkthrough moved to:
[dukersss13/arvee_web_ui/docs/application.md](https://github.com/dukersss13/arvee_web_ui/blob/main/docs/application.md)

## ArVee Agent

For a focused overview of RouterAgent + AgentTools architecture, clarification flow, and functionality, see [ArVee Agent details](md/arvee_agent.md).

### Custom Website UI
The web frontend has been split into a separate repository:
[dukersss13/arvee_web_ui](https://github.com/dukersss13/arvee_web_ui)

This backend repository now serves API endpoints only.

1. Install dependencies:
    `pip install -r requirements.txt`
2. Run the web app:
    `python3 webui/app.py`
3. Open your browser at:
    `http://localhost:7860`

The website supports session generation, uploading transaction/proof files, running validation, viewing results tables, and downloading validated CSV records.
Each session can be saved and loaded via a Session ID (`session_id`), and extracted transaction/proof inputs are persisted so previous sessions can be restored in the UI.

## Always-On Backend (Phase 1)

Initial production hardening is now included for container deployment.

1. Install/update dependencies:
    `pip install -r requirements.txt`
2. Start with Gunicorn:
    `gunicorn -c gunicorn.conf.py webui.app:app`

### Runtime Environment Variables

- `ARVEE_PORT` (default: `7860`)
- `ARVEE_HOST` (default: `0.0.0.0` for local launchers)
- `ARVEE_DEBUG` (default: `false`)
- `ARVEE_DB_URL` (optional, SQLAlchemy URL for remote DB; when unset uses local SQLite)
- `ARVEE_LOCAL_DB_NAME` (default: `receipt_validator_db`)
- `ARVEE_DB_ECHO` (default: `false`)
- `ARVEE_REQUIRE_USER_ID` (default: `false`; when `true`, requires `X-User-Id` header on session/validate endpoints)
- `GEMINI_API_KEY` (required for LLM calls)

### Docker Run

1. Build image:
    `docker build -t arvee-backend:latest .`
2. Run container:
    `docker run --rm -p 7860:7860 -e ARVEE_PORT=7860 arvee-backend:latest`

### Multi-User Header (Current Step)

For user-scoped session access, include an `X-User-Id` header in API requests.
If `ARVEE_REQUIRE_USER_ID=true`, requests without this header are rejected.


 ## 📌 TODO
Extend ArVee AgentTools Capability
