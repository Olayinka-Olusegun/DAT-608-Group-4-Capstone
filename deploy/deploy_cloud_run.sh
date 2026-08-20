#!/usr/bin/env bash
#
# Deploy the API and the dashboard to Cloud Run.
#
#   ./deploy/deploy_cloud_run.sh PROJECT_ID [REGION]
#
# Prerequisites, both of which are yours to do because they need your identity
# and your billing:
#
#   gcloud auth login
#   gcloud config set project PROJECT_ID
#
# The project must have billing enabled. Both services scale to zero, so the
# steady state cost when nobody is looking at them is the Artifact Registry
# storage for the images, which is a few cents a month.

set -euo pipefail

PROJECT="${1:-}"
REGION="${2:-europe-west1}"
REPO="pau-risk"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${PROJECT}" ]; then
  echo "usage: $0 PROJECT_ID [REGION]" >&2
  exit 2
fi

cd "${ROOT}"

log() { printf '\n=== %s\n' "$*"; }

log "Checking authentication"
if ! gcloud auth print-access-token >/dev/null 2>&1; then
  cat >&2 <<'MSG'
gcloud has no usable credentials. Run this yourself, then re-run this script:

  gcloud auth login

It opens a browser and signs you in. Nobody else can do this step for you.
MSG
  exit 1
fi

log "Enabling the services this deployment needs"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project "${PROJECT}"

log "Ensuring the Artifact Registry repository exists"
if ! gcloud artifacts repositories describe "${REPO}" \
      --location "${REGION}" --project "${PROJECT}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location "${REGION}" \
    --description "LGA violence risk service images" \
    --project "${PROJECT}"
fi

log "Building both images on Cloud Build"
gcloud builds submit \
  --config deploy/cloudbuild.yaml \
  --substitutions "_REGION=${REGION},_REPO=${REPO},_TAG=latest" \
  --project "${PROJECT}" \
  .

REGISTRY="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}"

log "Deploying the API"
# One CPU and 1 GiB is comfortable for reading the warehouse and running SHAP on
# a single week. Concurrency is left at the default because the work per request
# is small and mostly IO.
gcloud run deploy pau-risk-api \
  --image "${REGISTRY}/api:latest" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 1 --memory 1Gi \
  --min-instances 0 --max-instances 4 \
  --timeout 300 \
  --set-env-vars "LOG_LEVEL=INFO" \
  --project "${PROJECT}" \
  --quiet

API_URL="$(gcloud run services describe pau-risk-api \
  --region "${REGION}" --project "${PROJECT}" --format 'value(status.url)')"

log "Deploying the dashboard"
# Shiny holds a websocket per session, so session affinity is required or a
# reconnect can land on a different instance and lose the session. CPU boost
# shortens the cold start, which for an R process is otherwise noticeable.
gcloud run deploy pau-risk-dashboard \
  --image "${REGISTRY}/dashboard:latest" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 1 --memory 2Gi \
  --min-instances 0 --max-instances 4 \
  --concurrency 20 \
  --timeout 3600 \
  --session-affinity \
  --cpu-boost \
  --set-env-vars "PAU_RISK_ROOT=/app,PAU_RISK_API_URL=${API_URL}" \
  --project "${PROJECT}" \
  --quiet

APP_URL="$(gcloud run services describe pau-risk-dashboard \
  --region "${REGION}" --project "${PROJECT}" --format 'value(status.url)')"

log "Deployed"
printf 'Dashboard : %s\n' "${APP_URL}"
printf 'API       : %s\n' "${API_URL}"
printf 'API docs  : %s/docs\n' "${API_URL}"
printf 'Health    : %s/health\n' "${API_URL}"

log "Smoke test"
if curl -fsS --max-time 90 "${API_URL}/health" >/dev/null; then
  echo "API health check passed"
else
  echo "API health check did not pass, inspect the logs with:" >&2
  echo "  gcloud run services logs read pau-risk-api --region ${REGION} --project ${PROJECT}" >&2
fi
