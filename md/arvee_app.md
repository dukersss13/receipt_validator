# ArVee iOS App

ArVee is a native iOS companion to the Receipt Validator web application. It connects to the same Flask backend and provides the full validation + analytics pipeline from an iPhone or iPad.

### App Architecture Diagram

<img width="1020" height="1000" alt="ArVee iOS App Architecture" src="arvee_app_architecture.svg" />

---

## Architecture

The app is built with **SwiftUI** (iOS 17.0+, Swift 5.0, Xcode 15) and follows the **MVVM** pattern. Three shared `@StateObject` ViewModels are created at the root `MainTabView` and passed into child views.

| Layer | Components |
|-------|-----------|
| **Views** | MainTabView · HomeView · UploadView · ValidationView (Results) · ChatView · SettingsView · DiscrepancyCardView · RecommendationCardView · ManualMatchView · ResultCardView |
| **ViewModels** | SessionViewModel · ValidationViewModel · ChatViewModel |
| **Services** | APIService (HTTP singleton) · SSEClient (streaming) |
| **Models** | ChatMessage · ChatAskResponse · ChartData · ValidationResult · Session |
| **Theme** | ArVeeTheme (colors/fonts) · ArVeeStyle (view modifiers, button styles, StepIndicator) |

---

## How It Connects to the Backend

The iOS app communicates with the Flask backend over HTTP/JSON. The backend runs on port **7860** (default `http://localhost:7860`).

### Connection Bootstrap

1. On first launch, `ArVeeApp.init()` checks `UserDefaults` for a saved `apiBaseURL`.
2. If none exists, it writes `http://localhost:7860` as the default and sets `APIService.shared.baseURL`. The API service also supports LAN auto-discovery — scanning the local subnet for the backend.
3. The user can change the URL in the **Settings** tab — changes are persisted via `@AppStorage("apiBaseURL")` and immediately synced to `APIService.shared.baseURL`. Local HTTPS URLs are auto-normalized to HTTP.
4. A health check (`GET /api/health`) runs on the Settings screen to show a live connectivity indicator.
5. Sessions are created lazily — the Chat and Upload tabs call `ensureSession()` on appear, so users never need to manually create a session.

### API Endpoints Used

| iOS Method | HTTP | Flask Route | Purpose |
|-----------|------|-------------|---------|
| `healthCheck()` | GET | `/api/health` | Connectivity check → green/red indicator |
| `createSession()` | POST | `/api/session/new` | Creates a new session, returns `sessionId` |
| `loadSessionInputs(sessionId:)` | GET | `/api/session/<id>` | Loads transactions + proofs for a session |
| `validate(sessionId:...)` | POST | `/api/validate` | Multipart file upload + validation pipeline |
| `chatStreamURL(sessionId:message:)` | POST | `/api/chat/ask/stream` | SSE streaming chat with ArVee agent |
| `exportPDF(rows:)` | POST | `/api/export/validated` | Download validated results as PDF |

### Two Communication Modes

1. **Standard HTTP** — `APIService` (singleton) uses `URLSession` with 120s request / 300s resource timeouts. All endpoints except chat streaming use JSON request/response.

2. **Server-Sent Events (SSE)** — Chat uses `SSEClient`, a lightweight `URLSessionDataDelegate` that POSTs to `/api/chat/ask/stream` and receives a stream of events:
   ```
   event: token
   data: {"token": "Your"}

   event: token
   data: {"token": " total"}

   event: done
   data: {"answer": "...", "chart": {...}, "quickReplies": [...]}
   ```
   Tokens are appended to the assistant message in real-time, giving a typewriter effect. The `done` event carries the full response payload including chart data and quick reply suggestions.

---

## Tab-by-Tab Walkthrough

### 1. Home (Tab 0)
Landing screen with the ArVee brand and session status overview. The metric cards (Validated, Discrepancies, Unmatched) are tappable — tapping one switches to the Results tab and auto-scrolls to the corresponding section via a `resultsScrollTarget` binding.

### 2. Upload (Tab 1)
A 3-step guided flow: **Upload → Validate → Review**.

- **Upload**: Two file-picker cards — one for transactions (bank/card statements), one for proofs (receipt images). Supports both camera-roll photos via `PhotosPicker` and PDF/CSV documents via `fileImporter`.
- **Validate**: After selecting files, a "Validate" button triggers `SessionViewModel.ensureSession()` (auto-creates a session if needed), then `ValidationViewModel.loadAllFiles()` converts selected items into `FilePayload` structs, and `validate(sessionId:)` sends them as multipart form data to `POST /api/validate`.
- **Review**: On success, a summary card shows validated/discrepancy counts with a "View Results" button that switches to the Validation tab.

A `StepIndicator` at the top tracks progress with numbered circles and connecting line segments.

### 3. Results (Tab 2)
Displays validation results across five collapsible sections shown simultaneously (no tab switching):

| Section | Content | Interactive Actions |
|---------|---------|--------------------|
| Validated | Successfully matched transaction-receipt pairs | — |
| Discrepancies | Matched but with amount/date mismatches | Adjust amount, add comment, accept match |
| Unmatched Tx | Transactions with no matching receipt | Open Manual Match sheet |
| Unmatched Proofs | Receipts with no matching transaction | Open Manual Match sheet |
| Recommendations | AI-suggested matches with confidence scores | Accept individual or accept all |

A KPI row of gradient metric cards sits at the top. Each section has a collapsible header with item count badge and optional action button. Rows use `ResultCardView` — a compact card that shows matched-pair data (TX ↔ Proof) in a stacked layout.

**Interactive features:**
- **DiscrepancyCardView**: Shows TX vs Proof side-by-side with editable "Adjusted Amount" field and optional comment. "Accept Match" moves the item to Validated.
- **RecommendationCardView**: Shows the suggested match with reason/confidence. "Accept" moves it to Validated and removes matched items from unmatched lists. "Accept All" bulk-accepts.
- **ManualMatchView**: A sheet with selectable lists of unmatched transactions and proofs. Select one from each list and tap "Match Selected Pair" to create a validated row.

Supports deep-linking from the Home tab via `ScrollViewReader` — tapping a Home metric card scrolls directly to the corresponding section. Results can be exported as PDF via `POST /api/export/validated`.

### 4. Chat (Tab 3)
A native chat interface connected to the ArVee analytics agent. The chat tab automatically starts a session on appear via `.task { await sessionVM.ensureSession() }` — no manual session creation needed.

- **Welcome screen**: Shows when no messages have been sent. Displays the ArVee avatar with pill-style suggestion chips (e.g. "How much did I spend on food?", "Show my top 5 categories").
- **Chat bubbles**: User messages appear on the right in teal; assistant messages on the left in cream. Messages stream in real-time via SSE — tokens appear as they arrive from the backend.
- **Rich content**: Assistant messages can include inline charts (`ChartData`), category breakdowns (`topCategories`), and comparison tables (`comparisonTable`) — rendered as cards below the text.
- **Quick replies**: After an assistant response, if `quickReplies` are present in the payload, horizontal pill buttons appear for one-tap follow-up questions.
- **Streaming controls**: A stop button appears during streaming to cancel the SSE connection.

### 5. Settings (Tab 4)
Configuration screen with:

- **Backend URL**: Editable text field persisted to `@AppStorage("apiBaseURL")`. Changes take effect immediately.
- **Health check**: A live green/red/gray circle indicator showing backend connectivity. Runs automatically on appear and whenever the URL changes.
- **Version info**: App version display.

---

## Backend Pipeline (What Happens Server-Side)

When the iOS app sends requests, the Flask backend processes them through the same pipeline as the web UI:

### Validation Flow
```
POST /api/validate (multipart: sessionId + transaction files + proof files)
    ├── DataReader: parse files with Gemini Flash-Lite (async, parallel)
    │   ├── Images → base64 → Gemini VLM extraction
    │   ├── PDFs → PyPDF text → Gemini extraction
    │   └── CSVs → pandas DataFrame
    ├── Categorizer: assign spending categories (9 types)
    ├── Validator: fuzzy match + date/amount reconciliation
    │   ├── Validated transactions
    │   ├── Discrepancies (matched but mismatched)
    │   ├── Unmatched transactions + recommendations
    │   └── Unmatched proofs
    └── Save results to SQLite database
```

### Chat Flow
```
POST /api/chat/ask/stream (sessionId + message)
    ├── Load session state + validated rows from DB
    ├── RouterAgent: classify intent with Gemini (temp=0.0)
    │   ├── needs_clarification? → return question + quickReplies
    │   └── resolved → check QueryCache
    │       ├── cache hit → return cached result
    │       └── cache miss → AgentTools
    │           ├── spending_breakdown (single-period pandas aggregation)
    │           └── compare_spending_periods (two-period comparison)
    ├── Stream response as SSE events: token → token → ... → done
    └── Save chat turn to DB
```

---

## Design System

The app uses a warm "paper" theme inspired by Ramp's fintech aesthetic:

| Token | Hex | Usage |
|-------|-----|-------|
| Paper | `#F4EFE4` | Page backgrounds, tab bar |
| Sand | `#E9DBC1` | Card borders, dividers |
| Teal | `#0F7B6C` | Primary actions, accent color |
| Ink | `#1F1D1A` | Primary text |
| Coral | `#EA8F58` | Secondary highlights |
| Danger | `#B23A2C` | Errors, destructive actions |
| Success | `#34A853` | Confirmations, health check |

Custom view modifiers (`ArVeeStyle.swift`) provide consistent styling across the app — `.arveePageBackground()`, `ArveePrimaryButtonStyle()`, `ArveePillButtonStyle()`, `ArveeCard`, `ArveeMetricCard`, `StepIndicator`, and more.

---

## Running the App

### Prerequisites
- macOS with Xcode 15+
- Python 3.10+ with `requirements.txt` installed
- API keys configured in `secrets/`

### Steps
1. Start the backend:
   ```bash
   python backend.py
   ```
   This launches the Flask server on `http://localhost:7860` without opening a browser.

2. Open `arvee_app/ArVee.xcodeproj` in Xcode.

3. Select an iPhone simulator (or device) and build/run.

4. The app auto-connects to `localhost:7860`. If running on a physical device, update the backend URL in the Settings tab to your machine's local IP.
