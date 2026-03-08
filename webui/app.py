import io
import os
import tempfile
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from src.data.database import DataBase
from src.data.data_reader import DataReader, DataType
from src.validator import Validator
from src.utils.utils import create_session_id


app = Flask(__name__, template_folder="templates", static_folder="static")
database = DataBase(engine_name="receipt_validator_db", local_db=True)


def _save_uploaded_files(files: list[Any]) -> list[str]:
    """Persist uploaded files to a temporary location and return the temp paths."""
    temp_paths: list[str] = []

    for upload in files:
        suffix = os.path.splitext(upload.filename or "")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            upload.save(temp_file.name)
            temp_paths.append(temp_file.name)

    return temp_paths


def _cleanup_temp_files(file_paths: list[str]) -> None:
    """Best-effort cleanup for temporary files."""
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except OSError:
            pass


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    safe_frame = frame.where(pd.notna(frame), None)
    return safe_frame.to_dict(orient="records")


def _format_input_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []

    columns = ["business_name", "total", "date", "currency"]
    existing_columns = [col for col in columns if col in frame.columns]
    subset = frame[existing_columns].copy()

    if "date" in subset.columns:
        subset["date"] = subset["date"].astype(str)

    return _frame_to_records(subset)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/session/new")
def new_session():
    session_id = create_session_id()
    database.get_or_create_session(session_id)
    return jsonify({"sessionId": session_id})


@app.get("/api/session/<session_id>")
def get_session_inputs(session_id: str):
    try:
        transactions_df, proofs_df = database.load_session_history(session_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to load session: {exc}"}), 500

    return jsonify(
        {
            "sessionId": session_id,
            "transactions": _format_input_rows(transactions_df),
            "proofs": _format_input_rows(proofs_df),
        }
    )


@app.post("/api/validate")
def validate():
    session_id = str(request.form.get("sessionId", "")).strip()
    transactions = request.files.getlist("transactions")
    proofs = request.files.getlist("proofs")

    if not session_id:
        return (
            jsonify(
                {"error": "sessionId is required. Create or provide a session first."}
            ),
            400,
        )

    if not transactions:
        return jsonify({"error": "At least one transaction file is required."}), 400

    if not proofs:
        return jsonify({"error": "At least one proof image is required."}), 400

    transaction_paths = _save_uploaded_files(transactions)
    proof_paths = _save_uploaded_files(proofs)

    try:
        data_reader = DataReader(
            transactions=transaction_paths,
            proofs=proof_paths,
            database=database,
        )

        transactions_df = data_reader.load_data(DataType.TRANSACTIONS)
        proofs_df = data_reader.load_data(DataType.PROOFS)

        # Persist extracted user inputs by session_id for later retrieval.
        database.save_session_inputs(session_id, transactions_df, proofs_df)

        validator = Validator(transactions_df, proofs_df)
        results = validator.validate()
        summary_text, recommendations_df = validator.analyze_results(results)

        payload = {
            "sessionId": session_id,
            "summary": summary_text,
            "validatedTransactions": _frame_to_records(results.validated_transactions),
            "discrepancies": _frame_to_records(results.discrepancies),
            "unmatchedTransactions": _frame_to_records(results.unmatched_transactions),
            "unmatchedProofs": _frame_to_records(results.unmatched_proofs),
            "recommendations": _frame_to_records(recommendations_df),
        }

        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": f"Validation failed: {exc}"}), 500
    finally:
        _cleanup_temp_files(transaction_paths + proof_paths)


@app.post("/api/export/validated")
def export_validated():
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows", [])

    if not isinstance(rows, list) or not rows:
        return jsonify({"error": "No validated rows to export."}), 400

    frame = pd.DataFrame(rows)
    csv_buffer = io.StringIO()
    frame.to_csv(csv_buffer, index=False)

    binary_buffer = io.BytesIO(csv_buffer.getvalue().encode("utf-8"))
    binary_buffer.seek(0)

    return send_file(
        binary_buffer,
        as_attachment=True,
        download_name="validated_transactions.csv",
        mimetype="text/csv",
    )
