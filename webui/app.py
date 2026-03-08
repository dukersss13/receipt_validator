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


def _pdf_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_pdf_text(text: str, max_chars: int) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        lines.append(current)
        current = word

    lines.append(current)
    return lines


def _draw_text_block(
    ops: list[str],
    font_name: str,
    font_size: int,
    x: float,
    y: float,
    lines: list[str],
    line_height: float,
) -> None:
    if not lines:
        return

    # Ensure text is always rendered in solid black, independent of prior fill ops.
    ops.append("0 0 0 rg")
    ops.append("BT")
    ops.append(f"/{font_name} {font_size} Tf")
    ops.append(f"{x:.2f} {y:.2f} Td")
    ops.append(f"({_pdf_escape(lines[0])}) Tj")

    for line in lines[1:]:
        ops.append(f"0 -{line_height:.2f} Td ({_pdf_escape(line)}) Tj")

    ops.append("ET")


def _build_simple_table_pdf(frame: pd.DataFrame) -> bytes:
    """Build a one-page PDF with a formatted table for validated transactions."""
    columns = list(frame.columns)
    page_width = 842.0
    page_height = 595.0
    margin = 36.0
    table_width = page_width - (margin * 2)
    title_y = page_height - margin
    body_font_size = 8
    header_font_size = 9
    line_height = 10.0
    row_padding = 4.0
    min_row_height = line_height + (row_padding * 2)

    # Use content lengths to allocate readable column widths while keeping bounds.
    max_sample_rows = min(len(frame), 200)
    sample = frame.head(max_sample_rows)
    weights: list[float] = []
    for col in columns:
        sample_lengths = (
            sample[col].astype(str).map(len) if col in sample.columns else []
        )
        max_cell_len = max(sample_lengths) if len(sample_lengths) else 0
        weight = max(len(str(col)), min(max_cell_len, 36), 8)
        weights.append(float(weight))

    total_weight = sum(weights) or 1.0
    col_widths = [(weight / total_weight) * table_width for weight in weights]

    # Enforce a floor width and rebalance to exact table width.
    min_col_width = 70.0
    adjusted = [max(width, min_col_width) for width in col_widths]
    adjusted_total = sum(adjusted)
    scale = table_width / adjusted_total if adjusted_total else 1.0
    col_widths = [width * scale for width in adjusted]

    ops: list[str] = []

    _draw_text_block(
        ops,
        font_name="F2",
        font_size=14,
        x=margin,
        y=title_y,
        lines=["Validated Transactions"],
        line_height=14.0,
    )

    y_top = title_y - 24.0

    # Header cells.
    header_lines_per_col: list[list[str]] = []
    for idx, col in enumerate(columns):
        text_width = max(col_widths[idx] - (row_padding * 2), 10.0)
        max_chars = max(int(text_width / (header_font_size * 0.52)), 4)
        header_lines_per_col.append(_wrap_pdf_text(str(col), max_chars))

    header_row_lines = max((len(lines) for lines in header_lines_per_col), default=1)
    header_height = max(
        min_row_height, (header_row_lines * line_height) + (row_padding * 2)
    )

    # Draw header background and border.
    ops.append("0.93 0.95 0.98 rg")
    ops.append(
        f"{margin:.2f} {y_top - header_height:.2f} {table_width:.2f} {header_height:.2f} re f"
    )
    ops.append("0.25 0.25 0.25 RG")
    ops.append("0.8 w")
    ops.append(
        f"{margin:.2f} {y_top - header_height:.2f} {table_width:.2f} {header_height:.2f} re S"
    )

    current_x = margin
    for idx, lines in enumerate(header_lines_per_col):
        if idx > 0:
            ops.append(
                f"{current_x:.2f} {y_top - header_height:.2f} m {current_x:.2f} {y_top:.2f} l S"
            )
        text_x = current_x + row_padding
        text_y = y_top - row_padding - header_font_size
        _draw_text_block(
            ops,
            font_name="F2",
            font_size=header_font_size,
            x=text_x,
            y=text_y,
            lines=lines,
            line_height=line_height,
        )
        current_x += col_widths[idx]

    y_cursor = y_top - header_height
    max_table_bottom = margin
    rendered_rows = 0

    for _, row in frame.iterrows():
        wrapped_cells: list[list[str]] = []
        max_lines = 1

        for idx, col in enumerate(columns):
            value = row.get(col, "")
            if pd.isna(value):
                value = ""
            text_width = max(col_widths[idx] - (row_padding * 2), 10.0)
            max_chars = max(int(text_width / (body_font_size * 0.52)), 4)
            wrapped = _wrap_pdf_text(str(value), max_chars)
            wrapped_cells.append(wrapped)
            max_lines = max(max_lines, len(wrapped))

        row_height = max(min_row_height, (max_lines * line_height) + (row_padding * 2))
        if y_cursor - row_height < max_table_bottom:
            break

        ops.append("0.35 0.35 0.35 RG")
        ops.append("0.5 w")
        ops.append(
            f"{margin:.2f} {y_cursor - row_height:.2f} {table_width:.2f} {row_height:.2f} re S"
        )

        current_x = margin
        for idx, lines in enumerate(wrapped_cells):
            if idx > 0:
                ops.append(
                    f"{current_x:.2f} {y_cursor - row_height:.2f} m {current_x:.2f} {y_cursor:.2f} l S"
                )

            text_x = current_x + row_padding
            text_y = y_cursor - row_padding - body_font_size
            _draw_text_block(
                ops,
                font_name="F1",
                font_size=body_font_size,
                x=text_x,
                y=text_y,
                lines=lines,
                line_height=line_height,
            )
            current_x += col_widths[idx]

        y_cursor -= row_height
        rendered_rows += 1

    if rendered_rows < len(frame):
        remaining = len(frame) - rendered_rows
        _draw_text_block(
            ops,
            font_name="F1",
            font_size=9,
            x=margin,
            y=max(margin - 6, 24),
            lines=[
                f"Showing {rendered_rows} of {len(frame)} rows ({remaining} not shown)."
            ],
            line_height=10.0,
        )

    content_stream = "\n".join(ops).encode("utf-8")

    objects: list[bytes] = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 842 595] "
        b"/Resources << /Font << /F1 4 0 R /F2 6 0 R >> >> /Contents 5 0 R >> endobj\n"
    )
    objects.append(
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
    )
    objects.append(
        f"5 0 obj << /Length {len(content_stream)} >> stream\n".encode("utf-8")
        + content_stream
        + b"\nendstream endobj\n"
    )
    objects.append(
        b"6 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> endobj\n"
    )

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("utf-8"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("utf-8"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF"
        ).encode("utf-8")
    )
    return bytes(pdf)


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


@app.post("/api/session/<session_id>/save")
def save_session_state(session_id: str):
    payload = request.get_json(silent=True) or {}
    state = payload.get("state")

    if not isinstance(state, dict):
        return jsonify({"error": "state must be an object."}), 400

    try:
        database.save_session_state(session_id, state)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to save session: {exc}"}), 500

    return jsonify({"sessionId": session_id, "saved": True})


@app.get("/api/session/<session_id>/state")
def get_session_state(session_id: str):
    try:
        state = database.load_session_state(session_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to load session state: {exc}"}), 500

    return jsonify({"sessionId": session_id, "state": state})


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
        print(f"\n[Validation] Run started for session {session_id}\n")
        data_reader = DataReader(
            transactions=transaction_paths,
            proofs=proof_paths,
            database=database,
        )

        print("\n[Validation] Reading Transactions\n")
        transactions_df = data_reader.load_data(DataType.TRANSACTIONS)
        txn_cost = data_reader.get_ingestion_cost_summary()
        print(
            "\n[Validation] Reading Txn Cost: "
            f"${txn_cost['estimatedTotalCostUsd']:.2f} "
            f"({txn_cost['inputTokens']} in / {txn_cost['outputTokens']} out)"
        )

        print("\n[Validation] Reading Proofs\n")
        proofs_df = data_reader.load_data(DataType.PROOFS)
        total_cost = data_reader.get_ingestion_cost_summary()
        proof_cost = {
            "inputTokens": max(0, total_cost["inputTokens"] - txn_cost["inputTokens"]),
            "outputTokens": max(
                0, total_cost["outputTokens"] - txn_cost["outputTokens"]
            ),
            "estimatedTotalCostUsd": round(
                max(
                    0.0,
                    total_cost["estimatedTotalCostUsd"]
                    - txn_cost["estimatedTotalCostUsd"],
                ),
                2,
            ),
        }
        print(
            "\n[Validation] Reading Proofs Cost: "
            f"${proof_cost['estimatedTotalCostUsd']:.2f} "
            f"({proof_cost['inputTokens']} in / {proof_cost['outputTokens']} out)"
        )

        ingestion_cost = data_reader.log_ingestion_cost(session_id)

        # Persist extracted user inputs by session_id for later retrieval.
        database.save_session_inputs(session_id, transactions_df, proofs_df)

        validator = Validator(transactions_df, proofs_df)
        results = validator.validate()
        summary_text, recommendations_df = validator.analyze_results(results)

        payload = {
            "sessionId": session_id,
            "summary": summary_text,
            "ingestionCost": ingestion_cost,
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
    pdf_bytes = _build_simple_table_pdf(frame)
    binary_buffer = io.BytesIO(pdf_bytes)
    binary_buffer.seek(0)

    return send_file(
        binary_buffer,
        as_attachment=True,
        download_name="validated_transactions.pdf",
        mimetype="application/pdf",
    )
