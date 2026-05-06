# ArVee Agent

ArVee is the conversational analytics assistant in Receipt Validator. It answers user questions over validated transaction data — providing spending totals, averages, category breakdowns, top-N summaries, period-over-period comparisons, and interactive chart visualizations.

## Architecture

ArVee uses a two-stage **Router → Helper** agent architecture. A lightweight routing model classifies the user's intent and extracts structured parameters, then hands off execution to the HelperAgent which runs deterministic computations and synthesizes a natural-language answer.

### Component Overview

| Component | Role | Model |
|---|---|---|
| `RouterAgent` | Classifies intent, selects tool, extracts params as structured JSON | `gemini-2.0-flash-lite` (zero temperature) |
| `HelperAgent` | Executes the selected tool over transaction data, generates the final answer | `gemini-2.5-flash-lite` |
| `LLMBase` | Shared base class for Gemini config loading, API-key resolution, and model initialization | — |

### Request Flow

```
User Question
     │
     ▼
┌──────────────────┐
│   RouterAgent    │  Receives question + chat history
│  (gemini-2.0)    │  Returns JSON: { tool_name, tool_params, confidence }
└────────┬─────────┘
         │
         ▼
   ┌─────────────┐   needs_clarification = true?
   │  Clarify?   │──────────────────────────────────► Return clarification question to user
   └──────┬──────┘
          │ no
          ▼
┌──────────────────┐
│   HelperAgent    │  ask_with_routed_tool()
│  (gemini-2.5)    │  1. Execute tool (deterministic Python)
│                  │  2. Synthesize answer from tool output
│                  │  3. Attach chart payload if requested
└────────┬─────────┘
         │
         ▼
   Response payload
   (answer + chart + metadata)
```

### Key Design Decisions

- **Router is non-thinking**: Uses `gemini-2.0-flash-lite` (no chain-of-thought) with `temperature=0.0` and `max_tokens=150` for fast, deterministic JSON extraction.
- **Deterministic tool execution**: All math (filtering, grouping, aggregation) runs in Python via pandas — the LLM never computes numbers.
- **Chart payloads are data-only**: Tools return structured JSON chart descriptors (`{ type, labels, values, ... }`). The frontend renders them as inline SVG — no server-side image generation.
- **In-chat memory**: Prior turns are passed to both router and helper to support follow-up questions.

### ArVee Agent in the UI
<img width="1540" height="1137" alt="Screenshot 2026-04-27 at 9 48 51 PM" src="https://github.com/user-attachments/assets/1a6a409b-77bd-4dc1-95ab-19364f533eb6" />

### Architecture Image Placeholder

---

## Tools

ArVee exposes two tools that the RouterAgent selects between based on user intent.

### 1. `spending_breakdown`

Single-period aggregation over validated transactions. Filters by category (fuzzy-matched), time period, and aggregation method (`sum` or `average`). Supports top-N category ranking and optional bar or pie chart output.

### 2. `compare_spending_periods`

Compares spending between two periods (e.g. this month vs last month) with delta and percent-change computation. Supports weekly-average normalization and optional grouped-bar, bar, or pie chart output.

---

## Visualization

When the user requests a graph, the tool response includes a `chart` payload — a JSON descriptor that the frontend renders as inline SVG.

- **Bar / Pie** charts for single-period breakdowns via `spending_breakdown`
- **Grouped Bar** charts for period comparisons via `compare_spending_periods`
- Charts render above the text answer, with an expand button for full-screen viewing
- X-axis labels auto-rotate when there are more than 4 categories
- Text answers are limited to a top-5 summary when a chart is present

---

## Main Functionality

### 1. Spend Aggregation

Computes total spend for all validated rows or filtered subsets (for example, category or time-period queries). When no timeframe is specified, aggregates across all available transactions.

<img width="544" height="501" alt="Screenshot 2026-04-27 at 9 52 00 PM" src="https://github.com/user-attachments/assets/a0b2e865-8260-4d45-ac2b-88d019f67e27" />

### 2. Category Breakdown and Ranking

Returns top categories by sum or average using grouped aggregations across validated transactions.

<img width="547" height="504" alt="Screenshot 2026-04-27 at 9 52 13 PM" src="https://github.com/user-attachments/assets/390c12c0-f9ea-46c0-81e0-8cad8953309c" />

### 3. Time-Scoped Analysis

Supports time-period queries including "this month", "last month", "N months ago", and explicit `YYYY-MM` month references. Defaults to all available transactions when no timeframe is specified.

### 4. Period Comparison

Compares two periods side-by-side with delta and percent-change calculations. Supports weekly-average normalization for fair comparison of periods with different lengths.

### 5. Interactive Charts

Generates bar, pie, and grouped-bar chart visualizations rendered as inline SVG in the chat UI. Charts are expandable to full-screen for detailed inspection.
