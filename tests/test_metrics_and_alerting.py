"""The two places where an error would be invisible in ordinary use.

Precision at k is computed per week and then averaged, which is not the same as
computing it over the pooled test set, and the difference is easy to get wrong.
The alerting cooldown is tested because suppression bugs only show up weeks later,
as either a silent channel or a flood.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pau_risk.models.metrics import calibration_table, evaluate, precision_recall_at_k
from pau_risk.models.train import calibrate


def _frame(rows):
    return pd.DataFrame(rows, columns=["week_start", "lga_code", "score", "label"])


def test_precision_at_k_is_averaged_within_weeks():
    frame = _frame(
        [
            ("2024-01-01", "A", 0.9, 1), ("2024-01-01", "B", 0.8, 0),
            ("2024-01-01", "C", 0.1, 0), ("2024-01-01", "D", 0.05, 0),
            ("2024-01-08", "A", 0.9, 0), ("2024-01-08", "B", 0.8, 0),
            ("2024-01-08", "C", 0.7, 1), ("2024-01-08", "D", 0.05, 0),
        ]
    )
    scores = precision_recall_at_k(frame, k=2)
    # Week one catches its single positive in the top two, week two does not.
    assert scores["precision"] == pytest.approx((0.5 + 0.0) / 2)
    assert scores["recall"] == pytest.approx((1.0 + 0.0) / 2)


def test_lift_is_relative_to_the_weekly_base_rate():
    frame = _frame(
        [("2024-01-01", f"L{i}", 1.0 - i / 100, 1 if i == 0 else 0) for i in range(100)]
    )
    scores = precision_recall_at_k(frame, k=10)
    assert scores["precision"] == pytest.approx(0.1)
    assert scores["lift"] == pytest.approx(10.0)   # base rate is one in a hundred


def test_perfect_ranking_scores_one():
    frame = _frame(
        [
            ("2024-01-01", "A", 0.99, 1), ("2024-01-01", "B", 0.90, 1),
            ("2024-01-01", "C", 0.10, 0), ("2024-01-01", "D", 0.01, 0),
        ]
    )
    result = evaluate(frame, "perfect", k_values=(2,))
    assert result.roc_auc == pytest.approx(1.0)
    assert result.at_k[2]["recall"] == pytest.approx(1.0)


def test_calibration_table_reports_the_gap():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=2000)
    labels = (rng.uniform(size=2000) < scores).astype(int)
    table = calibration_table(labels, scores, bins=5)
    assert len(table) == 5
    assert table["gap"].abs().max() < 0.15


def test_tie_break_preserves_ordering_inside_a_plateau():
    from sklearn.isotonic import IsotonicRegression

    raw = np.array([0.10, 0.20, 0.30, 0.80, 0.90])
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw, np.array([0.0, 0.0, 0.0, 1.0, 1.0]))

    flat = calibrator.predict(raw)
    assert flat[0] == flat[1] == flat[2]        # the plateau isotonic produces

    adjusted = calibrate(calibrator, raw)
    assert adjusted[0] < adjusted[1] < adjusted[2]
    assert np.max(np.abs(adjusted - flat)) < 1e-5


# ------------------------------------------------------------------ alerting
@pytest.fixture
def warehouse(tmp_path, monkeypatch):
    from pau_risk import storage
    from pau_risk.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(
        type(settings), "sqlite_path", property(lambda self: tmp_path / "test.db")
    )
    monkeypatch.setattr(type(settings), "database_url", property(lambda self: None))
    storage.reset_engine()
    storage.init_schema(settings)
    yield settings
    storage.reset_engine()


def _predictions():
    return pd.DataFrame(
        [
            {"lga_code": "A", "lga_name": "Aville", "state_name": "Alpha",
             "risk_tier": "Severe", "probability": 0.4},
            {"lga_code": "B", "lga_name": "Bville", "state_name": "Alpha",
             "risk_tier": "High", "probability": 0.2},
            {"lga_code": "C", "lga_name": "Cville", "state_name": "Alpha",
             "risk_tier": "Low", "probability": 0.01},
        ]
    )


def test_only_tiers_above_the_threshold_alert(warehouse):
    from pau_risk.alerting import decide

    decisions = decide(_predictions(), warehouse)
    assert {d.lga_code for d in decisions} == {"A", "B"}
    assert all(d.action == "send" for d in decisions)


def test_repeat_at_the_same_tier_is_suppressed(warehouse):
    from datetime import date

    from pau_risk.alerting import decide, run

    predictions = _predictions()
    run("run-1", date(2024, 1, 1), predictions, send=False, settings=warehouse)
    # The first run was a dry run, so nothing is recorded as sent and nothing suppresses.
    assert all(d.action == "send" for d in decide(predictions, warehouse))

    from pau_risk.storage import upsert, now_iso

    upsert(
        "alerts",
        [
            {
                "alert_id": "prior:A", "run_id": "run-0", "lga_code": "A",
                "week_start": "2024-01-01", "risk_tier": "Severe", "probability": 0.4,
                "channel": "webhook", "status": "sent", "dispatched_at": now_iso(),
                "payload": "{}",
            }
        ],
    )
    decisions = {d.lga_code: d for d in decide(predictions, warehouse)}
    assert decisions["A"].action == "suppress_cooldown"
    assert decisions["B"].action == "send"


def test_escalation_overrides_the_cooldown(warehouse):
    from pau_risk.alerting import decide
    from pau_risk.storage import now_iso, upsert

    upsert(
        "alerts",
        [
            {
                "alert_id": "prior:A", "run_id": "run-0", "lga_code": "A",
                "week_start": "2024-01-01", "risk_tier": "High", "probability": 0.2,
                "channel": "webhook", "status": "sent", "dispatched_at": now_iso(),
                "payload": "{}",
            }
        ],
    )
    decisions = {d.lga_code: d for d in decide(_predictions(), warehouse)}
    assert decisions["A"].action == "send"
    assert "escalated" in decisions["A"].reason
