"""Alerting: turn a scored week into notifications, once.

The failure mode that destroys an early warning service is not a missed alert, it
is a stream of repeated ones. An LGA that stays severe for six weeks running
should generate one alert and then be visible on the dashboard, not six identical
messages, because a recipient who learns to ignore the channel cannot be reached
when something genuinely changes.

Three rules implement that. Only tiers at or above the configured minimum
generate an alert. An LGA that has already been alerted within the cooldown
window is suppressed unless its tier has risen, since an escalation is new
information even inside the window. And every decision, sent or suppressed, is
written to the alerts table, so the channel can be audited after the fact.

The job runs dry by default. Sending is an outward facing action, so it happens
only when the operator passes the flag explicitly and a destination is configured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Settings, get_settings
from ..logging_utils import get_logger
from ..storage import now_iso, read_sql, upsert

LOGGER = get_logger(__name__)

TIER_RANK = {"Low": 0, "Elevated": 1, "High": 2, "Severe": 3}


@dataclass
class AlertDecision:
    lga_code: str
    lga_name: str
    state_name: str
    risk_tier: str
    probability: float
    action: str          # send, suppress_cooldown or below_threshold
    reason: str

    def as_row(self, run_id: str, week_start: date, channel: str, status: str) -> dict[str, Any]:
        return {
            "alert_id": f"{run_id}:{self.lga_code}",
            "run_id": run_id,
            "lga_code": self.lga_code,
            "week_start": week_start.isoformat(),
            "risk_tier": self.risk_tier,
            "probability": float(self.probability),
            "channel": channel,
            "status": status,
            "dispatched_at": now_iso(),
            "payload": json.dumps(
                {
                    "lga": self.lga_name,
                    "state": self.state_name,
                    "action": self.action,
                    "reason": self.reason,
                }
            ),
        }


def _recent_alerts(cooldown_days: int) -> dict[str, tuple[str, str]]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=cooldown_days)).isoformat()
    frame = read_sql(
        """
        SELECT lga_code, risk_tier, dispatched_at
        FROM alerts
        WHERE status = 'sent' AND dispatched_at >= :cutoff
        ORDER BY dispatched_at DESC
        """,
        {"cutoff": cutoff},
    )
    history: dict[str, tuple[str, str]] = {}
    for row in frame.itertuples():
        history.setdefault(row.lga_code, (row.risk_tier, row.dispatched_at))
    return history


def decide(
    predictions: pd.DataFrame, settings: Settings | None = None
) -> list[AlertDecision]:
    settings = settings or get_settings()
    alert_cfg = settings.section("alerting")
    minimum = TIER_RANK[str(alert_cfg["minimum_tier"])]
    history = _recent_alerts(int(alert_cfg["cooldown_days"]))

    decisions: list[AlertDecision] = []
    for row in predictions.itertuples():
        tier_rank = TIER_RANK.get(str(row.risk_tier), 0)
        if tier_rank < minimum:
            continue
        previous = history.get(row.lga_code)
        if previous is not None:
            previous_rank = TIER_RANK.get(previous[0], 0)
            if tier_rank <= previous_rank:
                decisions.append(
                    AlertDecision(
                        lga_code=row.lga_code,
                        lga_name=row.lga_name,
                        state_name=row.state_name,
                        risk_tier=str(row.risk_tier),
                        probability=float(row.probability),
                        action="suppress_cooldown",
                        reason=f"already alerted at {previous[0]} on {previous[1][:10]}",
                    )
                )
                continue
            reason = f"escalated from {previous[0]} to {row.risk_tier}"
        else:
            reason = f"first {row.risk_tier} alert within the cooldown window"
        decisions.append(
            AlertDecision(
                lga_code=row.lga_code,
                lga_name=row.lga_name,
                state_name=row.state_name,
                risk_tier=str(row.risk_tier),
                probability=float(row.probability),
                action="send",
                reason=reason,
            )
        )
    return decisions


def _format_message(decisions: list[AlertDecision], week_start: date) -> str:
    lines = [f"LGA violence risk alert, week beginning {week_start.isoformat()}", ""]
    for decision in decisions:
        lines.append(
            f"{decision.lga_name}, {decision.state_name}: {decision.risk_tier} "
            f"({decision.probability:.3f}). {decision.reason}."
        )
    lines.append("")
    lines.append(
        "Probabilities are calibrated over a seven day horizon. This is a prioritisation "
        "aid, not a forecast of a specific incident."
    )
    return "\n".join(lines)


def _post_webhook(url: str, message: str) -> tuple[bool, str]:
    import requests

    try:
        response = requests.post(url, json={"text": message}, timeout=30)
        return response.status_code < 400, f"HTTP {response.status_code}"
    except requests.RequestException as exc:
        return False, exc.__class__.__name__


def run(
    run_id: str,
    week_start: date,
    predictions: pd.DataFrame,
    send: bool = False,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Evaluate the week, record every decision, and dispatch when asked to."""
    settings = settings or get_settings()
    alert_cfg = settings.section("alerting")
    decisions = decide(predictions, settings)
    to_send = [decision for decision in decisions if decision.action == "send"]

    webhook = settings.env(str(alert_cfg["webhook_url_env"]))
    dry_run = bool(alert_cfg["dry_run"]) and not send
    channel = "webhook" if webhook else "log"

    # Blast radius guard. In a normal week a handful of areas cross the
    # threshold. If a large share of the country does, the cause is far more
    # likely to be a broken upstream stage than a national emergency the model
    # alone has noticed, and the correct response is to hold the channel and make
    # a human look. This was added after an empty ingestion window produced a run
    # in which all 774 areas were scored severe.
    share = len(to_send) / max(len(predictions), 1)
    ceiling = float(alert_cfg.get("max_alert_share", 0.05))
    if share > ceiling:
        LOGGER.error(
            "holding alerts: %d of %d areas (%.0f%%) cleared the threshold, above the %.0f%% "
            "ceiling. This normally means the model or its inputs are broken, not that the "
            "country is. Inspect the run before dispatching.",
            len(to_send),
            len(predictions),
            100 * share,
            100 * ceiling,
        )
        dry_run = True
        for decision in to_send:
            decision.action = "held_blast_radius"
            decision.reason = f"held, {share:.0%} of areas above threshold"
        to_send = []

    status = "suppressed"
    detail = ""
    if not to_send:
        LOGGER.info("no LGA cleared the alerting threshold this week")
    elif dry_run:
        LOGGER.info(
            "dry run: %d alerts would be dispatched (%s)",
            len(to_send),
            ", ".join(f"{d.lga_name} {d.risk_tier}" for d in to_send[:5]),
        )
        status = "dry_run"
    elif webhook:
        delivered, detail = _post_webhook(webhook, _format_message(to_send, week_start))
        status = "sent" if delivered else "failed"
        LOGGER.info("webhook dispatch %s (%s)", status, detail)
    else:
        LOGGER.warning(
            "sending was requested but no destination is configured, recording as failed"
        )
        status = "failed"
        detail = "no ALERT_WEBHOOK_URL configured"

    rows = [
        decision.as_row(
            run_id,
            week_start,
            channel,
            status if decision.action == "send" else "suppressed",
        )
        for decision in decisions
    ]
    if rows:
        upsert("alerts", rows)

    frame = pd.DataFrame(
        [
            {
                "lga_code": decision.lga_code,
                "lga_name": decision.lga_name,
                "state_name": decision.state_name,
                "risk_tier": decision.risk_tier,
                "probability": decision.probability,
                "action": decision.action,
                "reason": decision.reason,
                "status": status if decision.action == "send" else "suppressed",
            }
            for decision in decisions
        ]
    )
    return frame


def recipients(settings: Settings | None = None) -> list[str]:
    settings = settings or get_settings()
    path = Path(settings.section("alerting")["recipients_file"])
    resolved = path if path.is_absolute() else settings.paths.root / path
    if not resolved.exists():
        return []
    return [
        line.strip()
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
