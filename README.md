# Receipt Validator

🧾 Transaction Validator System
A smart pipeline that extracts, validates, and reconciles financial transactions from statements and proof documents (PDFs or images), using LLMs for intelligent matching and discrepancy detection.


🔍 Overview
This project automates the validation of financial transactions by comparing user-submitted transaction statements with proof documents (e.g., receipts). It uses a hybrid system combining classic validation logic and the OpenAI API for fuzzy matching and recommendations.

🧠 Key Features
PDF/Image Ingestion: Reads transaction statements and proofs from PDFs or images.

LLM-Powered Extraction: Extracts merchant names, totals, and dates using OpenAI.

Smart Validation: Matches transactions and detects:

Full matches ✅

Discrepancies ⚠️

Unmatched transactions ❓

Recommendation Agent: Uses OpenAI to offer probable matches for unmatched entries (e.g., "Rolex" ⇌ "Watch Co.").

🔧 Components
1. Data Reader 📥
Loads documents using:

PyPDFLoader for PDFs

Custom payload extraction for images

Extracts raw transaction text

Sends to OpenAI for structured parsing

2. Validator ✅
Matches transactions between statements and proofs

Identifies and logs:

Discrepancies in totals

Unmatched or ambiguous entries

3. Discrepancy Calculator 📊
Highlights mismatched amounts (e.g., Lego: $34.00 ⇌ $36.00 → $2.00 discrepancy)

4. Recommendation Agent 🧠
Uses OpenAI to suggest closest matches for unmatched transactions
 
