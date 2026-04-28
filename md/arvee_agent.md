# ArVee HelperAgent

ArVee HelperAgent is the conversational assistant in Receipt Validator. It answers user questions over validated transaction data and is optimized for concise spending insights such as totals, averages, category breakdowns, and top-N summaries.

## Architecture

ArVee is built around a small, tool-calling agent architecture:

- LLM foundation via `LLMBase` for shared Gemini config loading, API-key resolution, and model initialization.
- `HelperAgent` orchestration for question handling, message construction, and response formatting.
- `breakdown_spending` tool for deterministic aggregation over validated transaction rows.
- Streaming + fallback flow: stream token output when available, then fall back to synchronous answer generation when needed.
- In-chat memory: prior turns are reused within the active chat to support follow-up questions.

### ArVee Agent in the UI
<img width="1540" height="1137" alt="Screenshot 2026-04-27 at 9 48 51 PM" src="https://github.com/user-attachments/assets/1a6a409b-77bd-4dc1-95ab-19364f533eb6" />


---

## Main Functionality

### 1. Spend Aggregation

Computes total spend for all validated rows or filtered subsets (for example, category or this-month queries).
<img width="544" height="501" alt="Screenshot 2026-04-27 at 9 52 00 PM" src="https://github.com/user-attachments/assets/a0b2e865-8260-4d45-ac2b-88d019f67e27" />



### 2. Category Breakdown and Ranking

Returns top categories by sum or average using grouped aggregations across validated transactions.

<img width="547" height="504" alt="Screenshot 2026-04-27 at 9 52 13 PM" src="https://github.com/user-attachments/assets/390c12c0-f9ea-46c0-81e0-8cad8953309c" />


### 3. Time-Scoped Analysis

Supports default month-scoped analysis and answers timeframe-aware spending questions when the user asks for trends or period summaries.


