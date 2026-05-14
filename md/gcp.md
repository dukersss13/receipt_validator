# GCP Backend: Current Runtime Snapshot

This document explains how the backend is currently running in GCP, based on the deployed Cloud Run service and current backend code paths.

Last verified: 2026-05-14

## Service Overview

- Service name: `arvee-backend`
- Project: `arvee-496304`
- Region: `us-central1`
- URL: `https://arvee-backend-5hqe7uiuka-uc.a.run.app`
- Latest ready revision: `arvee-backend-00010-f4z`
- Traffic split: `100%` to latest revision
- Runtime platform: Cloud Run (managed)

## Container + Scaling Runtime

From current Cloud Run service spec:

- Container concurrency: `20`
- Min instances: `1`
- Max instances: `20`
- Startup CPU boost: enabled
- Image source: Artifact Registry image digest (Cloud Run source deploy repository)

The service is always warm (minScale=1), so cold starts are reduced compared with scale-to-zero.

## Backend Process Model

Container startup command from `Dockerfile`:

- `gunicorn -c gunicorn.conf.py backend_app:app`

Gunicorn defaults from `gunicorn.conf.py`:

- Worker class: `gthread`
- Workers: `2` (env override: `ARVEE_GUNICORN_WORKERS`)
- Threads per worker: `8` (env override: `ARVEE_GUNICORN_THREADS`)
- Timeout: `180s`
- Graceful timeout: `30s`
- Keepalive: `5s`

## Secrets and Environment (Current Deployment)

Current Cloud Run env/secret wiring includes:

- `GEMINI_API_KEY` from Secret Manager (`gemini-api-key`)
- `EXCHANGE_RATE_KEY` from Secret Manager (`exchange-rate-key`)
- `ARVEE_GOOGLE_OAUTH_CLIENT_ID` from Secret Manager (`arvee-google-oauth-client-id`)
- `ARVEE_GOOGLE_OAUTH_IOS_CLIENT_ID` from Secret Manager (`arvee-google-oauth-ios-client-id`)
- `ARVEE_GOOGLE_REDIRECT_SCHEME=arvee`
- `ARVEE_AUTH_SECRET` set directly as an env value

Also configured:

- Secret volume mounted at `/app/secrets`
- Volume contains `exchange_rate_key` file from `exchange-rate-key` secret

Notes:

- `EXCHANGE_RATE_KEY` is currently available in both env and mounted file path.
- Google OAuth client ID is injected through env secret binding.

## Database Behavior (Important)

Code path in `backend_app.py`:

- If `ARVEE_DB_URL` is set: uses remote DB URL.
- If not set: falls back to local SQLite database (`receipt_validator_db`).

Current Cloud Run env does not show `ARVEE_DB_URL`, so the backend currently uses SQLite fallback inside the container filesystem.

Cloud Run is configured with a Cloud SQL instance annotation (`arvee-496304:us-central1:arvee-pg`), but without `ARVEE_DB_URL` the app does not actively use PostgreSQL.

Operational implication:

- SQLite on Cloud Run container filesystem is not durable across revision replacement and not ideal for multi-instance consistency.

## Auth and Identity Behavior

Auth is active and includes:

- Email/password endpoints
- Google token endpoint
- Signed access tokens backed by `ARVEE_AUTH_SECRET`

Relevant runtime defaults from code:

- `ARVEE_REQUIRE_USER_ID` defaults to `false` unless explicitly set
- If bearer token is present, backend resolves user from token
- If no bearer token and require-user-id is false, some flows can fallback to `anonymous`

## CORS Behavior

CORS logic is opt-in via `ARVEE_CORS_ORIGINS`.

- If `ARVEE_CORS_ORIGINS` is unset/empty, CORS headers are not added.
- If set, only configured origins are allowed (or wildcard if `*` is included).

## Health and Smoke Results (Current)

Observed live responses:

- `GET /api/health` -> `{ "status": "ok" }`
- `GET /api/health/deep` -> status `ok`, with:
  - database check `ok=true`
  - auth secret configured and non-default
  - Google OAuth enabled with redirect scheme `arvee`
- `POST /api/validate` with only `sessionId=smoke-session` returns HTTP 400:
  - `{"error":"Session 'smoke-session' not found"}`

Interpretation:

- Backend is reachable and healthy.
- Validation endpoint is enforcing session existence as expected.

## API Surface Served by Cloud Run Backend

Primary live routes include:

- Health: `/api/health`, `/api/health/deep`
- Auth: `/api/auth/signup`, `/api/auth/login`, `/api/auth/google/config`, `/api/auth/google/token`, `/api/auth/me`
- Session lifecycle: `/api/session/new`, `/api/session/<id>`, `/api/session/<id>/save`, `/api/session/<id>/state`
- Validation: `/api/validate`, `/api/validate/stream`
- Chat: `/api/chat/ask`, `/api/chat/ask/stream`
- Export: `/api/export/validated`

## Recommended Next Hardening Steps

1. Move production DB usage to Cloud SQL by setting `ARVEE_DB_URL` (currently missing).
2. Keep Cloud SQL annotation and `ARVEE_DB_URL` aligned to avoid accidental SQLite fallback.
3. Rotate and store `ARVEE_AUTH_SECRET` through Secret Manager instead of plain env value.
4. Set explicit `ARVEE_CORS_ORIGINS` for production frontend origins.
5. Add a small operational runbook for revision rollback and secret rotation.

## Quick Commands for Current State Checks

```bash
# Service shape

gcloud run services describe arvee-backend \
  --project arvee-496304 \
  --region us-central1 \
  --format='yaml(status.url,status.latestReadyRevisionName,status.traffic,spec.template.spec.containers[0].env,spec.template.spec.volumes,spec.template.spec.containers[0].volumeMounts,spec.template.spec.containerConcurrency,spec.template.metadata.annotations)'

# Health checks
curl -sS https://arvee-backend-5hqe7uiuka-uc.a.run.app/api/health
curl -sS https://arvee-backend-5hqe7uiuka-uc.a.run.app/api/health/deep
```
