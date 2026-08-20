#!/usr/bin/env bash
#
# Create a brand new GCP project and deploy both services into it.
#
#   ./deploy/create_and_deploy.sh [PROJECT_ID] [REGION]
#
# With no arguments it generates a project id of the form pau-risk-<date>-<rand>,
# which avoids collisions with the global project id namespace.
#
# You must run "gcloud auth login" first. That step opens a browser and signs you
# in with your Google account, so it is yours to do and cannot be automated.
#
# The script is idempotent: an existing project, repository or service is reused
# rather than recreated, so re-running after a failure resumes rather than
# duplicating.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PROJECT="${1:-pau-risk-$(date +%Y%m%d)-$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 4)}"
REGION="${2:-europe-west1}"

log() { printf '\n=== %s\n' "$*"; }
die() { printf '\n!!! %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
log "Checking authentication"
if ! gcloud auth print-access-token >/dev/null 2>&1; then
  cat >&2 <<'MSG'
gcloud has no usable credentials.

Run this yourself, then re-run this script:

  gcloud auth login

It opens a browser and signs you into Google. Nobody else can do this step for
you, and no automated path around it exists.
MSG
  exit 1
fi
ACCOUNT="$(gcloud config get-value account 2>/dev/null)"
log "Authenticated as ${ACCOUNT}"

# ------------------------------------------------------------------ billing
# Cloud Run and Cloud Build both require an OPEN billing account. Note that the
# "gcloud billing" command group is not present in every CLI build (it is absent
# from 442.0.0, which is what this was first run against), so the Cloud Billing
# REST API is used directly. It needs no extra components and behaves the same
# on every version.
log "Locating an open billing account"
TOKEN="$(gcloud auth print-access-token)"

BILLING="${PAU_RISK_BILLING_ACCOUNT:-}"
if [ -n "${BILLING}" ]; then
  BILLING="billingAccounts/${BILLING#billingAccounts/}"
else
  BILLING="$(curl -s -m 60 -H "Authorization: Bearer ${TOKEN}" \
    https://cloudbilling.googleapis.com/v1/billingAccounts \
    | python3 -c "
import json,sys
data = json.load(sys.stdin)
accounts = data.get('billingAccounts', [])
open_accounts = [a for a in accounts if a.get('open')]
print(open_accounts[0]['name'] if open_accounts else '')
print('CLOSED:' + ','.join(a['name'] for a in accounts if not a.get('open')), file=sys.stderr)
" 2>/dev/null)"
fi

if [ -z "${BILLING}" ]; then
  cat >&2 <<'MSG'
No OPEN billing account is available on this login.

A billing account can exist and still be closed, which is what blocks deployment:
Cloud Run, Cloud Build and Artifact Registry all refuse to enable without one.
The symptom further down would otherwise be:

  FAILED_PRECONDITION: Billing account for project 'N' is not open.

Reopen or create one here, which needs a valid payment method:

  https://console.cloud.google.com/billing

Then re-run this script. If you have several accounts, name the one to use:

  export PAU_RISK_BILLING_ACCOUNT=XXXXXX-XXXXXX-XXXXXX
MSG
  exit 1
fi
log "Using ${BILLING}"

# ------------------------------------------------------------------ project
if gcloud projects describe "${PROJECT}" >/dev/null 2>&1; then
  log "Project ${PROJECT} already exists, reusing it"
else
  log "Creating project ${PROJECT}"
  gcloud projects create "${PROJECT}" \
    --name="LGA Violence Risk Service" \
    || die "could not create ${PROJECT}. Project ids are globally unique, try another."
fi

log "Linking billing to ${PROJECT}"
LINKED="$(curl -s -m 90 -X PUT \
  -H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json" \
  -d "{\"billingAccountName\":\"${BILLING}\"}" \
  "https://cloudbilling.googleapis.com/v1/projects/${PROJECT}/billingInfo" \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('billingEnabled', False))")"

# Linking a closed account returns success with billingEnabled false, so the flag
# is checked rather than the exit status.
if [ "${LINKED}" != "True" ]; then
  die "billing did not enable on ${PROJECT}. The account ${BILLING} is closed. Reopen it at https://console.cloud.google.com/billing and re-run."
fi

gcloud config set project "${PROJECT}" >/dev/null

# The APIs take a moment to become usable after being enabled, and a build
# submitted too early fails with a confusing permission error rather than a
# clear "not enabled" one.
log "Enabling services"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project "${PROJECT}"

log "Waiting for the service enablement to propagate"
for _ in $(seq 1 30); do
  if gcloud services list --enabled --project "${PROJECT}" \
       --filter='config.name:run.googleapis.com' --format='value(config.name)' \
       2>/dev/null | grep -q run; then
    break
  fi
  sleep 5
done

# ------------------------------------------------------------------- deploy
log "Handing over to the deployment script"
exec "${ROOT}/deploy/deploy_cloud_run.sh" "${PROJECT}" "${REGION}"
