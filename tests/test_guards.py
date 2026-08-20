"""Regression tests for the failure found during the clean end-to-end run.

Running the pipeline against an empty warehouse with a short ingestion window
produced a panel with 484,524 rows and no positives. Training completed, every
metric was not a number, the scoring stage marked all 774 areas severe, and the
alerting stage prepared to dispatch all of them. Each of the three guards added
in response is tested here, because this is the class of bug that produces
confident output rather than an error, and it will not announce itself.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pau_risk.models.train import Split, _validate_split


def _fold(rows: int, positives: int, start: str) -> pd.DataFrame:
    weeks = pd.date_range(start, periods=rows, freq="W-MON")
    labels = [1] * positives + [0] * (rows - positives)
    return pd.DataFrame({"week_start": weeks, "label": labels})


def test_training_refuses_a_panel_with_no_positives():
    split = Split(
        train=_fold(500, 0, "2013-01-07"),
        valid=_fold(60, 0, "2022-01-03"),
        test=_fold(60, 0, "2023-01-02"),
    )
    with pytest.raises(ValueError, match="Refusing to train"):
        _validate_split(split)


def test_training_refuses_a_panel_with_too_few_positives():
    split = Split(
        train=_fold(500, 5, "2013-01-07"),
        valid=_fold(60, 30, "2022-01-03"),
        test=_fold(60, 30, "2023-01-02"),
    )
    with pytest.raises(ValueError, match="train fold has 5 positive weeks"):
        _validate_split(split)


def test_training_accepts_an_adequate_panel():
    split = Split(
        train=_fold(500, 300, "2013-01-07"),
        valid=_fold(60, 40, "2022-01-03"),
        test=_fold(60, 40, "2023-01-02"),
    )
    _validate_split(split)


def test_error_names_the_remedy():
    split = Split(
        train=_fold(10, 0, "2013-01-07"),
        valid=_fold(10, 0, "2022-01-03"),
        test=_fold(10, 0, "2023-01-02"),
    )
    with pytest.raises(ValueError) as raised:
        _validate_split(split)
    assert "ingest --since" in str(raised.value)


# ------------------------------------------------------------ blast radius
@pytest.fixture
def warehouse(tmp_path, monkeypatch):
    from pau_risk import storage
    from pau_risk.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(
        type(settings), "sqlite_path", property(lambda self: tmp_path / "guard.db")
    )
    monkeypatch.setattr(type(settings), "database_url", property(lambda self: None))
    storage.reset_engine()
    storage.init_schema(settings)
    yield settings
    storage.reset_engine()


def _all_severe(n: int = 774) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lga_code": [f"NG{i:06d}" for i in range(n)],
            "lga_name": [f"Area {i}" for i in range(n)],
            "state_name": ["Alpha"] * n,
            "risk_tier": ["Severe"] * n,
            "probability": [0.9] * n,
        }
    )


def test_a_whole_country_of_severe_areas_holds_the_channel(warehouse):
    from datetime import date

    from pau_risk.alerting import run

    decisions = run("run-x", date(2024, 1, 1), _all_severe(), send=True, settings=warehouse)
    assert (decisions["action"] == "held_blast_radius").all()
    assert (decisions["status"] == "suppressed").all()


def test_a_normal_week_still_dispatches(warehouse):
    from datetime import date

    from pau_risk.alerting import run

    frame = _all_severe()
    frame.loc[10:, "risk_tier"] = "Low"      # ten areas above the threshold
    frame.loc[10:, "probability"] = 0.01
    decisions = run("run-y", date(2024, 1, 1), frame, send=False, settings=warehouse)
    assert (decisions["action"] == "send").all()
    assert len(decisions) == 10
