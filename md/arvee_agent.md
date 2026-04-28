# ArVee HelperAgent

ArVee HelperAgent is the conversational assistant in Receipt Validator. It answers user questions over validated transaction data and is optimized for concise spending insights such as totals, averages, category breakdowns, and top-N summaries.

## Architecture

ArVee is built around a small, tool-calling agent architecture:

- LLM foundation via `LLMBase` for shared Gemini config loading, API-key resolution, and model initialization.
- `HelperAgent` orchestration for question handling, message construction, and response formatting.
- `breakdown_spending` tool for deterministic aggregation over validated transaction rows.
- Streaming + fallback flow: stream token output when available, then fall back to synchronous answer generation when needed.
- In-chat memory: prior turns are reused within the active chat to support follow-up questions.

### Architecture Image Placeholder

<!-- Insert architecture image here -->

---

## Main Functionality

### 1. Spend Aggregation

Computes total spend for all validated rows or filtered subsets (for example, category or this-month queries).

#### Image Placeholder

<!-- Insert spend aggregation image here -->

### 2. Category Breakdown and Ranking

Returns top categories by sum or average using grouped aggregations across validated transactions.

#### Image Placeholder

<!-- Insert category breakdown image here -->

### 3. Time-Scoped Analysis

Supports default month-scoped analysis and answers timeframe-aware spending questions when the user asks for trends or period summaries.

#### Image Placeholder

<!-- Insert time-scoped analysis image here -->

### 4. Conversational Follow-Ups

Uses chat history context to handle follow-up prompts without requiring the user to restate prior filters or topic.

#### Image Placeholder

<!-- Insert conversational follow-up image here -->

### 5. Robust Answer Delivery

Streams responses for better UX and falls back to a blocking answer path if streaming or tool events fail.

#### Image Placeholder

<!-- Insert robust answer delivery image here -->
