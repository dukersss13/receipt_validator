import io
import inspect
import json
import math
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import wraps
from queue import Queue
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd
from google.auth.transport import requests as google_auth_requests
from google.oauth2 import id_token as google_id_token
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from flask import Flask, Response, jsonify, request, send_file
from werkzeug.security import check_password_hash, generate_password_hash

from src.data.database import DataBase
from src.agents.router_agent import RouterAgent
from src.agents.validator import Validator
from src.utils.utils import create_session_id

app = Flask(__name__)


def _response_status_code(result: Any) -> int:
    """Best-effort status code extraction from Flask view return values."""
    if isinstance(result, Response):
        return int(result.status_code)

    if isinstance(result, tuple) and result:
        if len(result) >= 2 and isinstance(result[1], int):
            return int(result[1])
        first = result[0]
        if isinstance(first, Response):
            return int(first.status_code)

    return 200


def _response_error_class(result: Any) -> str:
    """Extract errorClass from a Flask view result when present."""
    try:
        response = app.make_response(result)
        payload = response.get_json(silent=True)
        if isinstance(payload, dict):
            value = str(payload.get("errorClass", "")).strip()
            if value:
                return value
    except Exception:
        pass
    return ""


def _auth_error_response(message: str, status_code: int, error_class: str):
    return jsonify({"error": message, "errorClass": error_class}), status_code


def _classify_auth_exception(exc: Exception) -> str:
    text = str(exc).strip().lower()
    if not text:
        return "internal_error"

    if "already exists" in text:
        return "email_conflict"
    if "email and password are required" in text:
        return "validation_error"
    if "valid email" in text or "at least 8 characters" in text:
        return "validation_error"
    if "idtoken is required" in text:
        return "validation_error"
    if "invalid email or password" in text:
        return "invalid_credentials"

    if "google oauth is not configured" in text:
        return "oauth_not_configured"
    if "invalid google" in text or "token issuer" in text or "not verified" in text:
        return "oauth_verify_failure"
    if "missing subject" in text or "missing a valid email" in text:
        return "oauth_verify_failure"

    if "token expired" in text:
        return "auth_token_expired"
    if "invalid authentication token" in text:
        return "auth_token_invalid"

    if "no such column" in text or "undefined column" in text or "provider_id" in text:
        return "schema_mismatch"

    if "timed out" in text or "timeout" in text:
        return "db_timeout"
    if "database" in text or "sql" in text or "connection" in text:
        return "database_error"

    return "internal_error"


def _signup_password_validation_error(password: str) -> str | None:
    if len(password) < 8:
        return "Password must be at least 8 characters."

    if not re.search(r"[A-Z]", password):
        return "Password must include at least 1 capital letter, 1 number, and 1 special character."

    if not re.search(r"\d", password):
        return "Password must include at least 1 capital letter, 1 number, and 1 special character."

    if not re.search(r"[^A-Za-z0-9]", password):
        return "Password must include at least 1 capital letter, 1 number, and 1 special character."

    return None


def _instrument_auth_endpoint(endpoint_name: str):
    """Log auth endpoint latency and status to make dependency stalls visible."""

    def _decorator(fn):
        @wraps(fn)
        def _wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except Exception:
                elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
                app.logger.exception(
                    "auth_request endpoint=%s status=500 duration_ms=%s outcome=exception",
                    endpoint_name,
                    elapsed_ms,
                )
                raise

            status_code = _response_status_code(result)
            error_class = _response_error_class(result)
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            outcome = "success"
            if status_code >= 500:
                outcome = "server_error"
            elif status_code >= 400:
                outcome = "client_error"

            app.logger.info(
                "auth_request endpoint=%s status=%s duration_ms=%s outcome=%s error_class=%s",
                endpoint_name,
                status_code,
                elapsed_ms,
                outcome,
                error_class,
            )
            return result

        return _wrapper

    return _decorator


def _read_secret_file(path: str) -> str:
    candidate = str(path or "").strip()
    if not candidate:
        return ""
    try:
        with open(candidate, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _resolve_secret(
    env_name: str,
    fallback_env_name: str,
    default_file_paths: tuple[str, ...] = (),
) -> str:
    direct = str(os.getenv(env_name, os.getenv(fallback_env_name, ""))).strip()
    if direct:
        return direct

    file_from_env = str(
        os.getenv(f"{env_name}_FILE", os.getenv(f"{fallback_env_name}_FILE", ""))
    ).strip()
    if file_from_env:
        value = _read_secret_file(file_from_env)
        if value:
            return value

    for file_path in default_file_paths:
        value = _read_secret_file(file_path)
        if value:
            return value

    return ""


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _request_user_id() -> str:
    """Resolve user identity from bearer token first, then fallback headers."""
    auth_header = str(request.headers.get("Authorization", "")).strip()
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            return _verify_access_token(token)

    user_id = str(request.headers.get("X-User-Id", "")).strip()
    if user_id:
        return user_id

    if _env_flag("ARVEE_REQUIRE_USER_ID", default=False):
        raise ValueError("Missing authentication. Provide bearer token or X-User-Id.")

    # Dev-friendly fallback for local testing when auth is not wired yet.
    return "anonymous"


def _is_auth_error(exc: ValueError) -> bool:
    text = str(exc).lower()
    return "authentication" in text or "x-user-id" in text or "token" in text


def _auth_serializer() -> URLSafeTimedSerializer:
    secret = str(os.getenv("ARVEE_AUTH_SECRET", "")).strip()
    if not secret:
        # Dev fallback only; set ARVEE_AUTH_SECRET in production.
        secret = "arvee-dev-auth-secret"
    return URLSafeTimedSerializer(secret_key=secret, salt="arvee-auth-v1")


def _create_access_token(user_id: str) -> str:
    payload = {
        "sub": str(user_id).strip().lower(),
        "iat": int(datetime.utcnow().timestamp()),
    }
    return _auth_serializer().dumps(payload)


def _verify_access_token(token: str) -> str:
    max_age_seconds = int(os.getenv("ARVEE_AUTH_TOKEN_MAX_AGE", "604800"))
    try:
        payload = _auth_serializer().loads(token, max_age=max_age_seconds)
    except SignatureExpired as exc:
        raise ValueError("Authentication token expired.") from exc
    except BadSignature as exc:
        raise ValueError("Invalid authentication token.") from exc

    user_id = str(payload.get("sub", "")).strip().lower()
    if not user_id:
        raise ValueError("Invalid authentication token payload.")
    return user_id


def _build_database() -> DataBase:
    db_url = str(os.getenv("ARVEE_DB_URL", "")).strip()
    db_echo = _env_flag("ARVEE_DB_ECHO", default=False)

    if db_url:
        return DataBase(engine_name=db_url, local_db=False, echo=db_echo)

    db_name = str(os.getenv("ARVEE_LOCAL_DB_NAME", "receipt_validator_db")).strip()
    if not db_name:
        db_name = "receipt_validator_db"
    return DataBase(engine_name=db_name, local_db=True, echo=db_echo)


database = _build_database()


def _parse_cors_origins() -> list[str]:
    raw = str(os.getenv("ARVEE_CORS_ORIGINS", "")).strip()
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _origin_allowed(origin: str, allow_list: list[str]) -> bool:
    if not origin:
        return False
    if "*" in allow_list:
        return True
    return origin in allow_list


def _api_base_url() -> str:
    configured = str(os.getenv("ARVEE_PUBLIC_BASE_URL", "")).strip()
    if configured:
        return configured.rstrip("/")
    return request.host_url.rstrip("/")


def _google_oauth_client_id() -> str:
    configured = _resolve_secret(
        "ARVEE_GOOGLE_OAUTH_CLIENT_ID",
        "GOOGLE_OAUTH_CLIENT_ID",
        default_file_paths=(
            "/secrets/google_oauth_client_id",
            "secrets/google_oauth_client_id",
        ),
    )
    return configured


def _google_oauth_redirect_scheme() -> str:
    configured = str(os.getenv("ARVEE_GOOGLE_REDIRECT_SCHEME", "arvee")).strip()
    return configured or "arvee"


def _verify_google_id_token(id_token: str) -> dict[str, Any]:
    token = str(id_token).strip()
    if not token:
        raise ValueError("idToken is required.")

    client_id = _google_oauth_client_id()
    if not client_id:
        raise ValueError("Google OAuth is not configured on this server.")

    try:
        claims = google_id_token.verify_oauth2_token(
            token,
            google_auth_requests.Request(),
            client_id,
        )
    except Exception as exc:
        raise ValueError("Invalid Google identity token.") from exc

    issuer = str(claims.get("iss", "")).strip().lower()
    if issuer not in {"accounts.google.com", "https://accounts.google.com"}:
        raise ValueError("Invalid Google token issuer.")

    email = str(claims.get("email", "")).strip().lower()
    if not email or "@" not in email:
        raise ValueError("Google token is missing a valid email.")

    if not bool(claims.get("email_verified", False)):
        raise ValueError("Google account email is not verified.")

    provider_id = str(claims.get("sub", "")).strip()
    if not provider_id:
        raise ValueError("Google token is missing subject.")

    return {
        "email": email,
        "providerId": provider_id,
        "name": str(claims.get("name", "")).strip() or None,
        "picture": str(claims.get("picture", "")).strip() or None,
    }


@app.after_request
def _apply_cors(response: Response) -> Response:
    allow_list = _parse_cors_origins()
    origin = str(request.headers.get("Origin", "")).strip()
    if allow_list and _origin_allowed(origin, allow_list):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = (
            "Authorization, Content-Type, X-User-Id"
        )
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/api/<path:_path>", methods=["OPTIONS"])
@app.route("/api", methods=["OPTIONS"])
def api_options(_path: str = "") -> Response:
    return Response(status=204)


def _pdf_escape(text: str) -> str:
    """Escape parentheses and backslashes for PDF string literals."""
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_pdf_text(text: str, max_chars: int) -> list[str]:
    """Word-wrap *text* into lines of at most *max_chars* characters."""
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
    """Append PDF text-rendering operators for a multi-line text block."""
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
    """Convert a DataFrame to a list of dicts, replacing NaN with None."""
    if frame is None or frame.empty:
        return []
    safe_frame = frame.where(pd.notna(frame), None)
    return safe_frame.to_dict(orient="records")


def _format_input_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Subset a DataFrame to display columns and convert to record dicts."""
    if frame is None or frame.empty:
        return []

    columns = [
        "business_name",
        "total",
        "date",
        "currency",
        "category",
    ]
    existing_columns = [col for col in columns if col in frame.columns]
    subset = frame[existing_columns].copy()

    if "date" in subset.columns:
        subset["date"] = subset["date"].astype(str)

    return _frame_to_records(subset)


def _records_to_input_frame(rows: Any) -> pd.DataFrame:
    """Convert API-provided row dicts into normalized input DataFrame shape."""
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame([], columns=["business_name", "total", "date", "currency"])

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame([], columns=["business_name", "total", "date", "currency"])

    # Accept either normalized (snake_case) or display-style column names.
    alias_map = {
        "Business Name": "business_name",
        "Total": "total",
        "Date": "date",
        "Currency": "currency",
    }
    frame = frame.rename(columns=alias_map)

    for column in ["business_name", "total", "date", "currency"]:
        if column not in frame.columns:
            frame[column] = None

    return frame[["business_name", "total", "date", "currency"]]


def _merge_ingestion_costs(costs: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum token counts and costs across multiple ingestion cost dicts."""
    if not costs:
        return {}

    models = [str(cost.get("model", "unknown")) for cost in costs if cost]
    merged = {
        "model": "+".join(models) if models else "unknown",
        "inputTokens": int(sum(int(cost.get("inputTokens", 0) or 0) for cost in costs)),
        "outputTokens": int(
            sum(int(cost.get("outputTokens", 0) or 0) for cost in costs)
        ),
        "llmCalls": int(sum(int(cost.get("llmCalls", 0) or 0) for cost in costs)),
        "standardCalls": int(
            sum(int(cost.get("standardCalls", 0) or 0) for cost in costs)
        ),
        "fxCalls": int(sum(int(cost.get("fxCalls", 0) or 0) for cost in costs)),
        "estimatedInputCostUsd": round(
            sum(float(cost.get("estimatedInputCostUsd", 0.0) or 0.0) for cost in costs),
            2,
        ),
        "estimatedOutputCostUsd": round(
            sum(
                float(cost.get("estimatedOutputCostUsd", 0.0) or 0.0) for cost in costs
            ),
            2,
        ),
        "estimatedTotalCostUsd": round(
            sum(float(cost.get("estimatedTotalCostUsd", 0.0) or 0.0) for cost in costs),
            2,
        ),
    }
    return merged


@app.get("/")
def index():
    """Service root: API metadata, or legacy web UI when explicitly enabled."""
    if _env_flag("ARVEE_ENABLE_LEGACY_WEB_UI", default=False):
        return jsonify(
            {
                "name": "ArVee Backend",
                "status": "ok",
                "mode": "legacy-web-ui",
                "message": "Legacy bundled web UI mode is deprecated.",
            }
        )

    return jsonify(
        {
            "name": "ArVee Backend",
            "status": "ok",
            "mode": "api-only",
            "frontend": {
                "separateRepo": "arvee_web_ui",
                "apiBaseUrl": _api_base_url(),
            },
        }
    )


@app.get("/api/health")
def health():
    """Return a simple health-check response."""
    return jsonify({"status": "ok"})


@app.get("/api/health/deep")
def health_deep():
    """Return dependency-aware health details used for production diagnostics."""
    started = time.perf_counter()

    db_check_started = time.perf_counter()
    db_ok = True
    db_error = ""
    try:
        database.get_user_auth("healthcheck@example.com")
    except Exception as exc:
        db_ok = False
        db_error = str(exc)
    db_duration_ms = round((time.perf_counter() - db_check_started) * 1000.0, 2)

    google_client_id = _google_oauth_client_id()
    google_enabled = bool(str(google_client_id).strip())

    auth_secret = str(os.getenv("ARVEE_AUTH_SECRET", "")).strip()
    auth_secret_configured = bool(auth_secret)
    auth_secret_is_default = auth_secret == "arvee-dev-auth-secret"

    checks = {
        "database": {
            "ok": db_ok,
            "durationMs": db_duration_ms,
            "error": db_error or None,
        },
        "authSecret": {
            "configured": auth_secret_configured,
            "usingDevDefault": auth_secret_is_default,
        },
        "googleOAuth": {
            "enabled": google_enabled,
            "redirectScheme": _google_oauth_redirect_scheme(),
        },
    }

    overall_ok = db_ok
    status_code = 200 if overall_ok else 503
    total_duration_ms = round((time.perf_counter() - started) * 1000.0, 2)

    if not overall_ok:
        app.logger.warning(
            "health_deep status=%s duration_ms=%s database_ok=%s auth_secret_configured=%s google_enabled=%s",
            status_code,
            total_duration_ms,
            db_ok,
            auth_secret_configured,
            google_enabled,
        )

    return (
        jsonify(
            {
                "status": "ok" if overall_ok else "degraded",
                "durationMs": total_duration_ms,
                "checks": checks,
            }
        ),
        status_code,
    )


@app.get("/api/meta")
def api_meta():
    """Return API metadata for standalone frontend clients."""
    return jsonify(
        {
            "service": "arvee-backend",
            "apiBaseUrl": _api_base_url(),
            "auth": {
                "methods": ["bearer-token", "x-user-id-legacy", "google-oauth"],
                "requireUserId": _env_flag("ARVEE_REQUIRE_USER_ID", default=False),
                "signupEndpoint": "/api/auth/signup",
                "loginEndpoint": "/api/auth/login",
                "googleTokenEndpoint": "/api/auth/google/token",
                "currentUserEndpoint": "/api/auth/me",
                "google": {
                    "enabled": bool(_google_oauth_client_id()),
                    "redirectScheme": _google_oauth_redirect_scheme(),
                },
            },
            "cors": {
                "configuredOrigins": _parse_cors_origins(),
            },
        }
    )


@app.post("/api/auth/signup")
@_instrument_auth_endpoint("signup")
def auth_signup():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))

    if not email or "@" not in email:
        return _auth_error_response(
            "A valid email is required.",
            400,
            "validation_error",
        )
    password_error = _signup_password_validation_error(password)
    if password_error:
        return _auth_error_response(
            password_error,
            400,
            "validation_error",
        )

    password_hash = generate_password_hash(password)

    try:
        user = database.create_user_auth(email, password_hash)
    except ValueError as exc:
        if "already exists" not in str(exc).lower():
            return _auth_error_response(str(exc), 400, _classify_auth_exception(exc))

        try:
            existing_user = database.get_user_auth(email)
        except ValueError:
            existing_user = None

        # If the email already belongs to a Google-linked account, allow signup
        # to set a real password so email/password login works as expected.
        if (
            existing_user is not None
            and str(existing_user.provider or "").lower() == "google"
        ):
            try:
                user = database.set_user_auth_password(email, password_hash)
            except ValueError as update_exc:
                return _auth_error_response(
                    str(update_exc),
                    400,
                    _classify_auth_exception(update_exc),
                )
            except Exception as update_exc:
                return _auth_error_response(
                    f"Failed to update account: {update_exc}",
                    500,
                    _classify_auth_exception(update_exc),
                )
        else:
            return _auth_error_response(str(exc), 409, _classify_auth_exception(exc))
    except Exception as exc:
        return _auth_error_response(
            f"Failed to create account: {exc}",
            500,
            _classify_auth_exception(exc),
        )

    token = _create_access_token(user.email)
    return jsonify(
        {
            "token": token,
            "user": {
                "email": user.email,
                "provider": str(user.provider or "email"),
            },
        }
    )


@app.post("/api/auth/login")
@_instrument_auth_endpoint("login")
def auth_login():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))

    if not email or not password:
        return _auth_error_response(
            "email and password are required.",
            400,
            "validation_error",
        )

    try:
        user = database.get_user_auth(email)
    except ValueError as exc:
        return _auth_error_response(str(exc), 400, _classify_auth_exception(exc))
    except Exception as exc:
        return _auth_error_response(str(exc), 500, _classify_auth_exception(exc))

    if user is None or not str(user.password_hash).strip():
        return _auth_error_response(
            "Invalid email or password.",
            401,
            "invalid_credentials",
        )

    if not check_password_hash(user.password_hash, password):
        return _auth_error_response(
            "Invalid email or password.",
            401,
            "invalid_credentials",
        )

    token = _create_access_token(user.email)
    return jsonify(
        {
            "token": token,
            "user": {
                "email": user.email,
                "provider": str(user.provider or "email"),
            },
        }
    )


@app.get("/api/auth/google/config")
def auth_google_config():
    client_id = _google_oauth_client_id()
    return jsonify(
        {
            "enabled": bool(client_id),
            "clientId": client_id,
            "redirectScheme": _google_oauth_redirect_scheme(),
        }
    )


@app.post("/api/auth/google/token")
@_instrument_auth_endpoint("google_token")
def auth_google_token():
    payload = request.get_json(silent=True) or {}
    id_token = str(payload.get("idToken", "")).strip()
    if not id_token:
        return _auth_error_response(
            "idToken is required.",
            400,
            "validation_error",
        )

    try:
        profile = _verify_google_id_token(id_token)
        fallback_hash = generate_password_hash(
            f"google-only:{profile['providerId']}:{uuid4().hex}"
        )
        user = database.create_or_link_google_user(
            email=profile["email"],
            provider_id=profile["providerId"],
            password_hash_fallback=fallback_hash,
        )
    except ValueError as exc:
        return _auth_error_response(str(exc), 400, _classify_auth_exception(exc))
    except Exception as exc:
        return _auth_error_response(
            f"Failed Google authentication: {exc}",
            500,
            _classify_auth_exception(exc),
        )

    token = _create_access_token(user.email)
    return jsonify(
        {
            "token": token,
            "user": {
                "email": user.email,
                "provider": "google",
                "name": profile.get("name"),
                "picture": profile.get("picture"),
            },
        }
    )


@app.get("/api/auth/me")
def auth_me():
    try:
        user_id = _request_user_id()
        user = database.get_user_auth(user_id)
        return jsonify(
            {
                "user": {
                    "id": user_id,
                    "email": user_id,
                    "provider": str(getattr(user, "provider", "email") or "email"),
                }
            }
        )
    except ValueError as exc:
        return _auth_error_response(str(exc), 401, _classify_auth_exception(exc))


@app.post("/api/session/new")
def new_session():
    """Create a new session and return its ID."""
    try:
        user_id = _request_user_id()
        session_id = create_session_id()
        database.get_or_create_session(session_id, user_id=user_id)
        return jsonify({"sessionId": session_id})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 401


@app.get("/api/session/<session_id>")
def get_session_inputs(session_id: str):
    """Return saved transactions and proofs for a session."""
    try:
        user_id = _request_user_id()
        transactions_df, proofs_df = database.load_session_history(
            session_id, user_id=user_id
        )
    except ValueError as exc:
        status = 401 if _is_auth_error(exc) else 404
        return jsonify({"error": str(exc)}), status
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
    """Persist the full UI state dict for a session."""
    payload = request.get_json(silent=True) or {}
    state = payload.get("state")

    if not isinstance(state, dict):
        return jsonify({"error": "state must be an object."}), 400

    try:
        user_id = _request_user_id()
        existing_state = database.load_session_state(session_id, user_id=user_id) or {}
        merged_state = {**existing_state, **state}

        transactions_rows = merged_state.get("loadedTransactions")
        proofs_rows = merged_state.get("loadedProofs")

        if isinstance(transactions_rows, list) and isinstance(proofs_rows, list):
            # Keep session inputs aligned with what the user saved in UI state.
            database.save_session_inputs(
                session_id,
                _records_to_input_frame(transactions_rows),
                _records_to_input_frame(proofs_rows),
                user_id=user_id,
            )

        database.save_session_state(session_id, merged_state, user_id=user_id)
    except ValueError as exc:
        status = 401 if _is_auth_error(exc) else 400
        return jsonify({"error": str(exc)}), status
    except Exception as exc:
        return jsonify({"error": f"Failed to save session: {exc}"}), 500

    return jsonify({"sessionId": session_id, "saved": True})


@app.get("/api/session/<session_id>/state")
def get_session_state(session_id: str):
    """Load and return the saved UI state for a session."""
    try:
        user_id = _request_user_id()
        state = database.load_session_state(session_id, user_id=user_id)
    except ValueError as exc:
        status = 401 if _is_auth_error(exc) else 404
        return jsonify({"error": str(exc)}), status
    except Exception as exc:
        return jsonify({"error": f"Failed to load session state: {exc}"}), 500

    return jsonify({"sessionId": session_id, "state": state})


def _run_validation_pipeline(
    session_id: str,
    user_id: str,
    transactions: list[Any],
    proofs: list[Any],
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Run validation and return the payload used by both JSON and SSE endpoints."""
    # Lazy import to avoid loading PDF/LLM parser stack during app startup.
    from src.data.data_reader import DataReader, DataType

    if not session_id:
        raise ValueError("sessionId is required. Create or provide a session first.")

    use_uploaded_files = bool(transactions or proofs)
    if use_uploaded_files and (not transactions or not proofs):
        raise ValueError(
            "Provide both transactions and proofs when uploading new files, "
            "or upload neither to use saved session inputs."
        )

    def emit(stage: str, percent: int) -> None:
        if progress_callback is None:
            return
        progress_callback(stage, percent)

    transaction_paths: list[str] = []
    proof_paths: list[str] = []
    if use_uploaded_files:
        emit("Preparing uploaded files...", 5)
        transaction_paths = _save_uploaded_files(transactions)
        proof_paths = _save_uploaded_files(proofs)

    try:
        print(f"\n[Validation] Run started for session {session_id}\n")
        ingestion_cost: dict[str, Any] = {}
        categorize_cost: dict[str, Any] = {}
        shared_config = DataReader._load_config_cached("config/config.conf")

        if use_uploaded_files:
            transactions_reader = DataReader(
                transactions=transaction_paths,
                proofs=proof_paths,
                database=database,
                parsed_config=shared_config,
            )
            proofs_reader = DataReader(
                transactions=transaction_paths,
                proofs=proof_paths,
                database=database,
                parsed_config=shared_config,
            )

            emit("Reading uploaded transactions and proofs...", 20)
            print("\n[Validation] Reading Transactions and Proofs in parallel\n")
            with ThreadPoolExecutor(max_workers=2) as executor:
                tx_future = executor.submit(
                    transactions_reader.load_data, DataType.TRANSACTIONS
                )
                proof_future = executor.submit(proofs_reader.load_data, DataType.PROOFS)
                transactions_df = tx_future.result()
                proofs_df = proof_future.result()

            emit("Computing ingestion cost summaries...", 40)
            txn_cost = transactions_reader.get_ingestion_cost_summary()
            print(
                "\n[Validation] Reading Txn Cost: "
                f"${txn_cost['estimatedTotalCostUsd']:.2f} "
                f"({txn_cost['inputTokens']} in / {txn_cost['outputTokens']} out)"
            )

            proofs_cost = proofs_reader.get_ingestion_cost_summary()
            proof_cost = {
                "inputTokens": int(proofs_cost["inputTokens"]),
                "outputTokens": int(proofs_cost["outputTokens"]),
                "estimatedTotalCostUsd": round(
                    float(proofs_cost["estimatedTotalCostUsd"]), 2
                ),
            }
            print(
                "\n[Validation] Reading Proofs Cost: "
                f"${proof_cost['estimatedTotalCostUsd']:.2f} "
                f"({proof_cost['inputTokens']} in / {proof_cost['outputTokens']} out)"
            )

            ingestion_cost = _merge_ingestion_costs([txn_cost, proofs_cost])

            log_entry = {
                "ts": pd.Timestamp.utcnow().isoformat(),
                "sessionId": session_id,
                "ingestion": ingestion_cost,
            }
            log_dir = transactions_reader.validated_data_path
            os.makedirs(log_dir, exist_ok=True)
            with open(
                os.path.join(log_dir, "ingestion_cost.log"), "a", encoding="utf-8"
            ) as log_file:
                log_file.write(json.dumps(log_entry) + "\n")

            print(f"\nIngestion usage: {log_entry}\n")

        else:
            emit("Loading saved session inputs...", 20)
            transactions_df, proofs_df = database.load_session_history(
                session_id, user_id=user_id
            )
            if transactions_df.empty or proofs_df.empty:
                raise ValueError(
                    "No saved inputs found for this session. "
                    "Upload transactions and proofs first."
                )

        emit("Validating transactions against proofs...", 60)
        validator = Validator(
            transactions_df,
            proofs_df,
            parsed_config=shared_config,
        )
        results = validator.validate()

        emit("Building summary and recommendations...", 78)
        summary_text, recommendations_df = validator.analyze_results(results)
        categorize_cost = validator.categorize_cost
        enriched_transactions_df = validator.transactions
        enriched_proofs_df = validator.proofs

        print(
            "\n[Validation] Categorize Cost: "
            f"${float(categorize_cost.get('estimatedTotalCostUsd', 0.0)):.6f} "
            f"({int(categorize_cost.get('inputTokens', 0))} in / "
            f"{int(categorize_cost.get('outputTokens', 0))} out), "
            f"latency={float(categorize_cost.get('latencySeconds', 0.0)):.3f}s\n"
        )

        log_dir = str(shared_config.get("data_path.validated", "data/validated"))
        os.makedirs(log_dir, exist_ok=True)
        with open(
            os.path.join(log_dir, "categorize_cost.log"), "a", encoding="utf-8"
        ) as log_file:
            log_file.write(
                json.dumps(
                    {
                        "ts": pd.Timestamp.utcnow().isoformat(),
                        "sessionId": session_id,
                        "categorize": categorize_cost,
                    }
                )
                + "\n"
            )

        if use_uploaded_files:
            # Persist canonical extracted inputs in DB; categorization is preserved in session state.
            database.save_session_inputs(
                session_id, transactions_df, proofs_df, user_id=user_id
            )

        payload = {
            "sessionId": session_id,
            "summary": summary_text,
            "ingestionCost": ingestion_cost,
            "categorizeCost": categorize_cost,
            "transactions": _format_input_rows(enriched_transactions_df),
            "proofs": _format_input_rows(enriched_proofs_df),
            "validatedTransactions": _frame_to_records(results.validated_transactions),
            "discrepancies": _frame_to_records(results.discrepancies),
            "unmatchedTransactions": _frame_to_records(results.unmatched_transactions),
            "unmatchedProofs": _frame_to_records(results.unmatched_proofs),
            "recommendations": _frame_to_records(recommendations_df),
        }

        emit("Saving validation results...", 90)

        # Auto-save full session state after each successful validation run.
        existing_state = database.load_session_state(session_id, user_id=user_id) or {}
        database.save_session_state(
            session_id,
            {
                "summary": summary_text,
                "categorizeCost": categorize_cost,
                "loadedTransactions": payload["transactions"],
                "loadedProofs": payload["proofs"],
                "validatedTransactions": payload["validatedTransactions"],
                "discrepancies": payload["discrepancies"],
                "unmatchedTransactions": payload["unmatchedTransactions"],
                "unmatchedProofs": payload["unmatchedProofs"],
                "recommendations": payload["recommendations"],
                "chatHistory": existing_state.get("chatHistory", []),
            },
            user_id=user_id,
        )

        emit("Validation complete.", 100)
        return payload
    finally:
        _cleanup_temp_files(transaction_paths + proof_paths)


def _call_validation_pipeline(
    session_id: str,
    user_id: str,
    transactions: list[Any],
    proofs: list[Any],
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Call pipeline in a backward-compatible way for monkeypatched test stubs."""
    params = inspect.signature(_run_validation_pipeline).parameters
    if "user_id" in params:
        return _run_validation_pipeline(
            session_id=session_id,
            user_id=user_id,
            transactions=transactions,
            proofs=proofs,
            progress_callback=progress_callback,
        )

    return _run_validation_pipeline(
        session_id=session_id,
        transactions=transactions,
        proofs=proofs,
        progress_callback=progress_callback,
    )


@app.post("/api/validate")
def validate():
    """Run the full validation pipeline on uploaded or saved session data."""
    session_id = str(request.form.get("sessionId", "")).strip()
    transactions = request.files.getlist("transactions")
    proofs = request.files.getlist("proofs")

    try:
        user_id = _request_user_id()
        payload = _call_validation_pipeline(
            session_id=session_id,
            user_id=user_id,
            transactions=transactions,
            proofs=proofs,
        )
        return jsonify(payload)
    except ValueError as exc:
        status = 401 if _is_auth_error(exc) else 400
        return jsonify({"error": str(exc)}), status
    except Exception as exc:
        return jsonify({"error": f"Validation failed: {exc}"}), 500


@app.post("/api/validate/stream")
def validate_stream():
    """Stream validation progress events and final payload over SSE."""
    session_id = str(request.form.get("sessionId", "")).strip()
    transactions = request.files.getlist("transactions")
    proofs = request.files.getlist("proofs")

    if not session_id:
        return jsonify({"error": "sessionId is required."}), 400

    try:
        user_id = _request_user_id()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 401

    def generate() -> Any:
        yield _sse("start", {"sessionId": session_id})

        queue: Queue[tuple[str, dict[str, Any]]] = Queue()

        def on_progress(stage: str, percent: int) -> None:
            queue.put(("progress", {"stage": stage, "percent": percent}))

        def worker() -> None:
            try:
                payload = _call_validation_pipeline(
                    session_id=session_id,
                    user_id=user_id,
                    transactions=transactions,
                    proofs=proofs,
                    progress_callback=on_progress,
                )
                queue.put(("done", payload))
            except ValueError as exc:
                queue.put(("error", {"error": str(exc)}))
            except Exception as exc:
                queue.put(("error", {"error": f"Validation failed: {exc}"}))
            finally:
                queue.put(("end", {}))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            event, payload = queue.get()
            if event == "progress":
                yield _sse("progress", payload)
                continue
            if event == "done":
                yield _sse("done", payload)
                continue
            if event == "error":
                yield _sse("error", payload)
                continue
            if event == "end":
                break

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/export/validated")
def export_validated():
    """Export validated transaction rows as a PDF file."""
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


@app.post("/api/chat/ask")
def chat_ask():
    """Handle a single chat question and return the full response."""
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("sessionId", "")).strip()
    message = str(payload.get("message", "")).strip()

    if not session_id:
        return jsonify({"error": "sessionId is required."}), 400

    if not message:
        return jsonify({"error": "message is required."}), 400

    try:
        state = database.load_session_state(session_id) or {}
        validated_rows = state.get("validatedTransactions", [])

        if not isinstance(validated_rows, list) or not validated_rows:
            guidance = _validation_required_chat_payload(message)
            return jsonify({"sessionId": session_id, "question": message, **guidance})

        router = RouterAgent()
        chat_history = state.get("chatHistory", [])
        if not isinstance(chat_history, list):
            chat_history = []
        result = router.ask(
            message,
            validated_rows,
            chat_history=chat_history,
        )
        chat_history.extend(
            [
                {"role": "user", "text": message, "ts": datetime.utcnow().isoformat()},
                {
                    "role": "assistant",
                    "text": result.get("answer", ""),
                    "ts": datetime.utcnow().isoformat(),
                },
            ]
        )
        state["chatHistory"] = chat_history
        database.save_session_state(session_id, state)
        return jsonify({"sessionId": session_id, "question": message, **result})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Chat failed: {exc}"}), 500


def _sse(event: str, payload: dict[str, Any]) -> str:
    """Format a Server-Sent Event message string."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _stream_token_chunks(
    text: str,
    target_chunk_chars: int = 22,
    max_chunks: int = 160,
) -> list[str]:
    """Split answer text into readable chunks for incremental SSE token streaming."""
    normalized = str(text or "")
    parts = re.findall(r"\S+\s*", normalized)
    if not parts:
        return [normalized] if normalized else []

    chunks: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) + len(part) > max(8, target_chunk_chars):
            chunks.append(current)
            current = part
        else:
            current += part

    if current:
        chunks.append(current)

    if len(chunks) > max_chunks:
        group_size = max(1, math.ceil(len(chunks) / max_chunks))
        chunks = [
            "".join(chunks[idx : idx + group_size])
            for idx in range(0, len(chunks), group_size)
        ]

    return [chunk for chunk in chunks if chunk]


def _validation_required_chat_payload(question: str = "") -> dict[str, Any]:
    """Build a friendly chat response when validation data is missing."""
    normalized = str(question or "").strip().lower()

    if "what should i do after upload" in normalized:
        answer = "Run validation and wait for the results"
    else:
        answer = (
            "Please upload your transactions and proofs in the Upload tab, "
            "then validate before asking questions"
        )

    return {
        "answer": answer,
        "rowsScanned": 0,
        "confidence": "high",
        "toolUsed": False,
        "toolName": "",
        "route": "validation_required",
        "needsClarification": True,
        "quickReplies": [
            "How do I upload Transactions/Proofs?",
            "What should I do after upload?",
        ],
        "chart": None,
        "top_categories": [],
        "comparison_table": None,
    }


@app.post("/api/chat/ask/stream")
def chat_ask_stream():
    """Stream a chat response as Server-Sent Events."""
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("sessionId", "")).strip()
    message = str(payload.get("message", "")).strip()

    if not session_id:
        return jsonify({"error": "sessionId is required."}), 400

    if not message:
        return jsonify({"error": "message is required."}), 400

    state = database.load_session_state(session_id) or {}
    validated_rows = state.get("validatedTransactions", [])

    if not isinstance(validated_rows, list) or not validated_rows:
        guidance = _validation_required_chat_payload(message)

        def generate_validation_guidance() -> Any:
            yield _sse("start", {"sessionId": session_id})
            yield _sse(
                "progress",
                {
                    "stage": "Validation is needed before chat can answer this.",
                    "percent": 100,
                },
            )
            yield _sse("token", {"token": guidance["answer"]})
            yield _sse("done", guidance)

        return Response(
            generate_validation_guidance(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    router = RouterAgent()
    chat_history = state.get("chatHistory", [])
    if not isinstance(chat_history, list):
        chat_history = []

    def generate() -> Any:
        """Yield SSE frames for the streamed chat response."""
        streamed_chunks: list[str] = []
        try:
            yield _sse("start", {"sessionId": session_id})
            yield _sse(
                "progress", {"stage": "Looking into your request...", "percent": 15}
            )
            result = router.ask(
                message,
                validated_rows,
                chat_history=chat_history,
            )
            yield _sse(
                "progress",
                {
                    "stage": "Analyzing your validated transactions...",
                    "percent": 65,
                },
            )
            final_answer = str(result.get("answer", "") or "").strip()
            if not final_answer:
                final_answer = "I could not generate an answer."

            yield _sse(
                "progress", {"stage": "Finalizing the response...", "percent": 90}
            )
            for token_chunk in _stream_token_chunks(final_answer):
                streamed_chunks.append(token_chunk)
                yield _sse("token", {"token": token_chunk})

            assembled_answer = "".join(streamed_chunks)
            if assembled_answer:
                final_answer = assembled_answer.strip() or final_answer

            chat_history.extend(
                [
                    {
                        "role": "user",
                        "text": message,
                        "ts": datetime.utcnow().isoformat(),
                    },
                    {
                        "role": "assistant",
                        "text": final_answer,
                        "ts": datetime.utcnow().isoformat(),
                    },
                ]
            )
            state["chatHistory"] = chat_history
            database.save_session_state(session_id, state)

            yield _sse("progress", {"stage": "Done.", "percent": 100})
            yield _sse(
                "done",
                {
                    "answer": final_answer,
                    "rowsScanned": len(validated_rows),
                    "confidence": result.get("confidence", "high"),
                    "toolUsed": bool(result.get("toolUsed", False)),
                    "toolName": result.get("toolName", ""),
                    "needsClarification": bool(result.get("needsClarification", False)),
                    "quickReplies": result.get("quickReplies", []),
                    "chart": result.get("chart"),
                    "top_categories": result.get("top_categories"),
                    "comparison_table": (
                        result["chart"].get("table")
                        if isinstance(result.get("chart"), dict)
                        else None
                    ),
                },
            )
        except Exception as exc:
            yield _sse("error", {"error": f"Chat failed: {exc}"})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
