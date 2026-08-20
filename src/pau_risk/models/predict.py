"""Score the forecast week and write the results the service layer reads.

One scoring run produces four things that belong together and are written under a
single run identifier: the run's own metadata and metrics, a calibrated
probability with a tier and two rankings for every one of the 774 LGAs, the top
drivers behind each of those scores, and the threat actor graph for the same
window. Keying all of them by run identifier means the dashboard, the brief and
the alert job are always describing the same model, and an audit of a decision
taken in a particular week can reconstruct exactly what the council was shown.

Every LGA is scored, including the ones that have never recorded an event. A list
that silently omits quiet areas would be worse than useless for the problem in
the brief, which is precisely that risk migrates into places that were recently
quiet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

import pandas as pd

from ..config import Settings, get_settings
from ..logging_utils import get_logger
from ..storage import now_iso, upsert
from .explain import Explanation, explain
from .train import TrainedModel

LOGGER = get_logger(__name__)


@dataclass
class ScoringRun:
    run_id: str
    week_start: date
    predictions: pd.DataFrame
    explanation: Explanation

    @property
    def top(self) -> pd.DataFrame:
        return self.predictions.head(10)


def score_week(
    model: TrainedModel,
    panel: pd.DataFrame,
    week_start: pd.Timestamp | None = None,
    settings: Settings | None = None,
) -> ScoringRun:
    settings = settings or get_settings()
    week_start = week_start or panel["week_start"].max()
    frame = panel[panel["week_start"] == week_start].copy()
    if frame.empty:
        raise ValueError(f"no panel rows for week {week_start}")

    frame["probability"] = model.score(frame)
    frame["risk_tier"] = model.tier(frame["probability"].to_numpy())
    frame = frame.sort_values("probability", ascending=False).reset_index(drop=True)
    frame["rank_national"] = range(1, len(frame) + 1)
    frame["rank_state"] = frame.groupby("state_name")["probability"].rank(
        ascending=False, method="first"
    ).astype(int)

    explanation = explain(model.booster, frame, model.features, top_k=5)

    LOGGER.info(
        "scored week %s: %s",
        week_start.date(),
        ", ".join(
            f"{tier}={count}"
            for tier, count in frame["risk_tier"].value_counts().items()
        ),
    )
    return ScoringRun(
        run_id=model.run_id,
        week_start=week_start.date(),
        predictions=frame,
        explanation=explanation,
    )


def persist(run: ScoringRun, model: TrainedModel, notes: str = "") -> dict[str, int]:
    """Write the run, its predictions and its drivers to the warehouse."""
    upsert(
        "model_runs",
        [
            {
                "run_id": run.run_id,
                "created_at": now_iso(),
                "model_version": model.model_version,
                "horizon_days": 7,
                "train_end": model.train_end.isoformat(),
                "metrics": json.dumps(model.metrics, default=str),
                "notes": notes,
            }
        ],
    )

    prediction_rows = [
        {
            "run_id": run.run_id,
            "lga_code": row.lga_code,
            "week_start": run.week_start.isoformat(),
            "probability": float(row.probability),
            "risk_tier": str(row.risk_tier),
            "rank_national": int(row.rank_national),
            "rank_state": int(row.rank_state),
        }
        for row in run.predictions.itertuples()
    ]
    written_predictions = upsert("predictions", prediction_rows)

    drivers = run.explanation.drivers.copy()
    drivers["run_id"] = run.run_id
    drivers["week_start"] = run.week_start.isoformat()
    written_drivers = upsert(
        "prediction_drivers",
        drivers[
            [
                "run_id", "lga_code", "week_start", "driver_rank",
                "feature_name", "feature_label", "feature_value", "shap_value",
            ]
        ].to_dict(orient="records"),
    )

    LOGGER.info(
        "persisted run %s: %d predictions, %d drivers",
        run.run_id,
        written_predictions,
        written_drivers,
    )
    return {"predictions": written_predictions, "drivers": written_drivers}
