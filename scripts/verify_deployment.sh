#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <service_url> [--require-google]"
  exit 2
fi

SERVICE_URL="${1%/}"
REQUIRE_GOOGLE="false"
if [[ ${2:-} == "--require-google" ]]; then
  REQUIRE_GOOGLE="true"
fi

fetch_json() {
  local url="$1"
  curl -fsS "$url"
}

json_query() {
  local query="$1"
  python3 -c "import json,sys; data=json.load(sys.stdin); print(${query})"
}

echo "[verify] Checking shallow health..."
HEALTH_JSON="$(fetch_json "${SERVICE_URL}/api/health")"
HEALTH_STATUS="$(printf '%s' "$HEALTH_JSON" | json_query 'repr(data.get("status"))')"
if [[ "$HEALTH_STATUS" != "'ok'" ]]; then
  echo "[verify] FAIL /api/health status=${HEALTH_STATUS}"
  exit 1
fi

echo "[verify] Checking deep health..."
DEEP_JSON="$(fetch_json "${SERVICE_URL}/api/health/deep")"
DB_OK="$(printf '%s' "$DEEP_JSON" | json_query 'str(bool(data.get("checks",{}).get("database",{}).get("ok")))')"
EPHEMERAL_RISK="$(printf '%s' "$DEEP_JSON" | json_query 'str(bool(data.get("checks",{}).get("database",{}).get("runtime",{}).get("isEphemeralRisk")))')"
AUTH_SECRET_DEFAULT="$(printf '%s' "$DEEP_JSON" | json_query 'str(bool(data.get("checks",{}).get("authSecret",{}).get("usingDevDefault")))')"

if [[ "$DB_OK" != "True" ]]; then
  echo "[verify] FAIL deep health: database.ok is not true"
  exit 1
fi

if [[ "$EPHEMERAL_RISK" == "True" ]]; then
  echo "[verify] FAIL deep health: database runtime indicates sqlite-local ephemeral risk"
  exit 1
fi

if [[ "$AUTH_SECRET_DEFAULT" == "True" ]]; then
  echo "[verify] FAIL deep health: auth secret is using development default"
  exit 1
fi

echo "[verify] Checking Google config endpoint..."
GOOGLE_JSON="$(fetch_json "${SERVICE_URL}/api/auth/google/config")"
GOOGLE_ENABLED="$(printf '%s' "$GOOGLE_JSON" | json_query 'str(bool(data.get("enabled")))')"
if [[ "$REQUIRE_GOOGLE" == "true" && "$GOOGLE_ENABLED" != "True" ]]; then
  echo "[verify] FAIL Google OAuth required but /api/auth/google/config is disabled"
  exit 1
fi

echo "[verify] PASS"
