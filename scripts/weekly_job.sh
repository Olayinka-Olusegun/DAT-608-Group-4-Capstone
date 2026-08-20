#!/usr/bin/env bash
#
# Weekly refresh for the LGA violence risk service.
#
# Scheduled for early Monday so that a scored week starts on the Monday it
# describes. Install with:
#
#   crontab -e
#   5 4 * * 1 /path/to/DAT\ 608/scripts/weekly_job.sh >> /var/log/pau-risk.log 2>&1
#
# The stages are separate commands rather than one pipeline call so that a
# failure in, say, the brief does not prevent the alerts from going out: the
# scored predictions are already in the warehouse by then, and each later stage
# reads from there rather than from the stage before it.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
cd "${ROOT}" || exit 1

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

run_stage() {
  local name="$1"; shift
  log "stage ${name} starting"
  if "$@"; then
    log "stage ${name} ok"
    return 0
  fi
  log "stage ${name} FAILED with status $?"
  return 1
}

# Refresh the feeds and rebuild the panel. A failure here leaves last week's
# predictions in place, which is the correct degradation: a stale score is more
# useful than no score, provided the dashboard reports the run date, which it does.
run_stage ingest   "${PYTHON}" -m pau_risk.cli ingest --days 14 || exit 1
run_stage features "${PYTHON}" -m pau_risk.cli features || exit 1

# The model is retrained monthly rather than weekly. Weekly retraining on a
# panel that grows by 774 rows a week changes nothing about the fit and makes
# week to week movements in the score impossible to attribute, because the model
# and the data would both have moved.
if [ "$(date +%d)" -le 7 ]; then
  run_stage train "${PYTHON}" -m pau_risk.cli train || log "training failed, scoring with the existing model"
fi

run_stage score "${PYTHON}" -m pau_risk.cli score --notes "weekly scheduled run" || exit 1
run_stage brief "${PYTHON}" -m pau_risk.cli brief || log "brief generation failed, continuing to alerts"

# Dispatch is opt in. Set PAU_RISK_SEND_ALERTS=1 in the cron environment once a
# destination is configured and the recipient list has been confirmed.
if [ "${PAU_RISK_SEND_ALERTS:-0}" = "1" ]; then
  run_stage alert "${PYTHON}" -m pau_risk.cli alert --send
else
  run_stage alert "${PYTHON}" -m pau_risk.cli alert
  log "alerts evaluated in dry run mode, set PAU_RISK_SEND_ALERTS=1 to dispatch"
fi

log "weekly job complete"
