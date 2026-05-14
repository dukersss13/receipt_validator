# GCP Setup Guide for ArVee

This document explains how to set up Google Cloud Platform (GCP) for this repository, including:

- Required GCP services/APIs
- Secret setup
- Container build and deploy to Cloud Run
- Database options
- How to connect this GitHub repo to GCP for CI/CD

## 1. Prerequisites

Install and verify:

- Google Cloud CLI (gcloud)
- Docker
- GitHub access to this repository

Authenticate:

```bash
gcloud auth login
gcloud auth application-default login
```

Set variables for your environment:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export SERVICE="arvee-backend"
export REPO="arvee"
export IMAGE="arvee-backend"

gcloud config set project "$PROJECT_ID"
```

## 2. Create or Select a GCP Project

If needed, create a project:

```bash
gcloud projects create "$PROJECT_ID" --name="ArVee"
```

Link billing (required for Cloud Run, Artifact Registry, and most APIs):

```bash
gcloud beta billing projects link "$PROJECT_ID" --billing-account="YOUR_BILLING_ACCOUNT_ID"
```

## 3. Enable Required APIs

Enable all core services used by deployment/runtime:

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com
```

Optional (only if using Cloud SQL PostgreSQL):

```bash
gcloud services enable \
  sqladmin.googleapis.com \
  servicenetworking.googleapis.com
```

## 4. Create Artifact Registry

Create a Docker repository once per region:

```bash
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="ArVee container images"
```

Configure Docker authentication:

```bash
gcloud auth configure-docker "$REGION-docker.pkg.dev"
```

## 5. Secrets: What This Repo Needs

This app reads Gemini credentials and exchange-rate credentials. In local development, secret files are read from the secrets directory. In GCP, use Secret Manager and inject as environment variables.

### Required for production

- GEMINI_API_KEY (or GOOGLE_API_KEY): required by the LLM layer
- EXCHANGE_RATE_KEY: used for non-USD currency conversion

### Optional runtime env vars

- ARVEE_DB_URL: SQLAlchemy URL for remote DB (recommended for production)
- ARVEE_REQUIRE_USER_ID: set true to enforce X-User-Id header
- ARVEE_DB_ECHO: SQL logging toggle (usually false)

Create secrets:

```bash
printf '%s' 'YOUR_GEMINI_API_KEY' | gcloud secrets create GEMINI_API_KEY --data-file=-
printf '%s' 'YOUR_EXCHANGE_RATE_KEY' | gcloud secrets create EXCHANGE_RATE_KEY --data-file=-
```

If secrets already exist, add new versions:

```bash
printf '%s' 'NEW_VALUE' | gcloud secrets versions add GEMINI_API_KEY --data-file=-
printf '%s' 'NEW_VALUE' | gcloud secrets versions add EXCHANGE_RATE_KEY --data-file=-
```

Grant Cloud Run runtime service account access (replace SA if needed):

```bash
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding EXCHANGE_RATE_KEY \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/secretmanager.secretAccessor"
```

## 6. Build and Push Container

From repo root:

```bash
docker build -t "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:latest" .
docker push "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:latest"
```

Alternative using Cloud Build:

```bash
gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:latest" .
```

## 7. Deploy to Cloud Run

Deploy with secret injection and production-friendly settings:

```bash
gcloud run deploy "$SERVICE" \
  --image "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:latest" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --port 7860 \
  --set-env-vars ARVEE_PORT=7860,ARVEE_HOST=0.0.0.0,ARVEE_DEBUG=false,ARVEE_REQUIRE_USER_ID=false,ARVEE_PUBLIC_BASE_URL=https://YOUR_SERVICE_URL,ARVEE_CORS_ORIGINS=* \
  --set-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest,EXCHANGE_RATE_KEY=EXCHANGE_RATE_KEY:latest
```

Important: Cloud Run also sets PORT at runtime. This app uses ARVEE_PORT, so we explicitly set ARVEE_PORT=7860 and deploy with --port 7860 to match container behavior.

Get service URL:

```bash
gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'

Run health checks (replace URL):

```bash
curl -sS https://YOUR_SERVICE_URL/api/health | jq .
curl -sS https://YOUR_SERVICE_URL/api/health/deep | jq .
curl -sS https://YOUR_SERVICE_URL/api/auth/google/config | jq .
```

`/api/health/deep` should report `status: "ok"` and `checks.database.ok: true` before iOS signup/Google auth tests.

Cloud Run stale revision guard:

```bash
gcloud run services describe "$SERVICE" --region "$REGION" \
  --format='value(status.latestCreatedRevisionName,status.latestReadyRevisionName,status.traffic[0].revisionName,status.traffic[0].percent)'
```

`latestCreatedRevisionName`, `latestReadyRevisionName`, and active traffic revision should match before iOS validation/chat tests.
```

## 8. Database Options

### Option A: Keep SQLite (quick start, not recommended for scaling)

If ARVEE_DB_URL is unset, the app uses a local SQLite DB file in the container filesystem. This is ephemeral and not suitable for durable multi-instance production.

### Option B: Cloud SQL PostgreSQL (recommended)

1. Create Cloud SQL PostgreSQL instance.
2. Create DB and user.
3. Build SQLAlchemy URL as ARVEE_DB_URL, for example:

```text
postgresql+psycopg://DB_USER:DB_PASSWORD@/DB_NAME?host=/cloudsql/PROJECT:REGION:INSTANCE
```

4. Deploy Cloud Run with:

- --add-cloudsql-instances PROJECT:REGION:INSTANCE
- ARVEE_DB_URL set as env var or Secret Manager secret

Example deploy add-on:

```bash
gcloud run services update "$SERVICE" \
  --region "$REGION" \
  --add-cloudsql-instances "PROJECT:REGION:INSTANCE" \
  --set-env-vars "ARVEE_DB_URL=postgresql+psycopg://DB_USER:DB_PASSWORD@/DB_NAME?host=/cloudsql/PROJECT:REGION:INSTANCE"
```

## 9. Connect GCP to This GitHub Repo

You have two common approaches.

### Approach A: Cloud Build trigger from GitHub

1. In GCP Console: Cloud Build -> Triggers -> Connect Repository.
2. Install the Cloud Build GitHub App.
3. Select this repository and branch (for example main).
4. Trigger type: push to branch.
5. Build config: Dockerfile in repo root.
6. Configure deploy step (Cloud Run) in trigger settings or cloudbuild.yaml.

Minimal cloudbuild.yaml example:

```yaml
steps:
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - -t
      - ${_IMAGE}
      - .
  - name: gcr.io/cloud-builders/docker
    args:
      - push
      - ${_IMAGE}
  - name: gcr.io/google.com/cloudsdktool/cloud-sdk
    entrypoint: gcloud
    args:
      - run
      - deploy
      - ${_SERVICE}
      - --image
      - ${_IMAGE}
      - --region
      - ${_REGION}
      - --platform
      - managed
      - --allow-unauthenticated
      - --port
      - '7860'
      - --set-env-vars
      - ARVEE_PORT=7860,ARVEE_HOST=0.0.0.0,ARVEE_DEBUG=false
      - --set-secrets
      - GEMINI_API_KEY=GEMINI_API_KEY:latest,EXCHANGE_RATE_KEY=EXCHANGE_RATE_KEY:latest
substitutions:
  _REGION: us-central1
  _SERVICE: arvee-backend
  _IMAGE: us-central1-docker.pkg.dev/$PROJECT_ID/arvee/arvee-backend:latest
images:
  - ${_IMAGE}
```

### Approach B: GitHub Actions with Workload Identity Federation

Recommended when you want deployment automation managed in GitHub.

1. Create Workload Identity Pool + Provider in GCP.
2. Create a deploy service account.
3. Grant roles:
   - roles/run.admin
   - roles/artifactregistry.writer
   - roles/iam.serviceAccountUser
   - roles/secretmanager.secretAccessor
4. Trust your GitHub repo in provider attribute conditions.
5. Use google-github-actions/auth and google-github-actions/deploy-cloudrun in a workflow.

This avoids storing long-lived GCP JSON keys in GitHub secrets.

## 10. Repo Runtime Mapping (Quick Reference)

- Entry web service: backend_app.py (served by Gunicorn via Docker CMD)
- Container port: 7860
- Runtime env toggles:
  - ARVEE_PORT, ARVEE_HOST, ARVEE_DEBUG
  - ARVEE_DB_URL, ARVEE_LOCAL_DB_NAME, ARVEE_DB_ECHO
  - ARVEE_REQUIRE_USER_ID
  - GEMINI_API_KEY or GOOGLE_API_KEY

## 11. Post-Deploy Validation Checklist

1. Open Cloud Run URL and verify index page loads.
2. Upload sample transactions/proofs and run validation.
3. Confirm logs show successful LLM calls (no missing API key errors).
4. Confirm non-USD conversion succeeds (exchange key wired).
5. If ARVEE_REQUIRE_USER_ID=true, verify API requests include X-User-Id.
6. If using Cloud SQL, verify session data persists across revisions/restarts.

## 12. Troubleshooting

- Error: Missing GEMINI_API_KEY or GOOGLE_API_KEY
  - Ensure secret exists, is bound to runtime service account, and mapped with --set-secrets.
- Conversion returns -1 for foreign currency
  - Validate EXCHANGE_RATE_KEY value and provider quota/status.
- Session state disappears after deploy/restart
  - You are likely on SQLite in ephemeral filesystem; move to Cloud SQL and set ARVEE_DB_URL.
- 401 Missing X-User-Id header
  - Disable ARVEE_REQUIRE_USER_ID or send X-User-Id from clients.

---

If you want, I can also add a production-ready cloudbuild.yaml and a GitHub Actions workflow file tailored to this exact repository.
