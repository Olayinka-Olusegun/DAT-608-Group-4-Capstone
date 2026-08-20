"""Leakage is the failure this suite exists to prevent.

Every feature on a row for week W must be computable on the Sunday before W. If
any window silently includes W itself, the model learns to read the answer, the
reported metrics become fiction, and nothing about the code looks wrong. The
tests below construct a panel with events at known weeks and assert the exact
values the windows should take, which is the only way to catch an off by one that
still produces plausible numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pau_risk.features.panel import _prefix, _weeks_since, _window, build_panel


def test_window_excludes_the_current_week():
    counts = np.array([[0.0, 1.0, 0.0, 0.0, 2.0, 0.0]])
    prefix = _prefix(counts)
    one_week = _window(prefix, 1)
    # Week index 2 must see the event in week 1 and nothing from week 2 onward.
    assert one_week[0, 0] == 0.0
    assert one_week[0, 1] == 0.0
    assert one_week[0, 2] == 1.0
    assert one_week[0, 5] == 2.0
    four_week = _window(prefix, 4)
    assert four_week[0, 4] == 1.0     # the event in week 1 only
    assert four_week[0, 5] == 3.0     # weeks 1 through 4


def test_weeks_since_is_censored_and_causal():
    counts = np.array([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
    gaps = _weeks_since(counts, censor=10)
    assert gaps[0, 0] == 10.0         # nothing has happened yet
    assert gaps[0, 1] == 10.0         # the event in week 1 is not visible at week 1
    assert gaps[0, 2] == 1.0
    assert gaps[0, 4] == 3.0
    assert gaps[0, 5] == 1.0


@pytest.fixture
def toy_inputs():
    registry = pd.DataFrame(
        [
            {"lga_code": "A", "lga_name": "Aville", "state_code": "S1", "state_name": "Alpha",
             "zone": "North West", "senatorial_district": None, "area_sqkm": 100.0,
             "centre_lat": 11.0, "centre_lon": 7.0},
            {"lga_code": "B", "lga_name": "Bville", "state_code": "S1", "state_name": "Alpha",
             "zone": "North West", "senatorial_district": None, "area_sqkm": 120.0,
             "centre_lat": 11.2, "centre_lon": 7.2},
        ]
    )
    adjacency = pd.DataFrame(
        [
            {"lga_code": "A", "neighbour_code": "B", "weight": 1.0, "centroid_km": 25.0},
            {"lga_code": "B", "neighbour_code": "A", "weight": 1.0, "centroid_km": 25.0},
        ]
    )
    dates = pd.to_datetime(["2013-03-04", "2013-03-11", "2016-06-06", "2019-09-02", "2022-04-04"])
    incidents = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(len(dates))],
            "event_date": dates,
            "event_class": ["banditry_kidnapping"] * 4 + ["state_operation"],
            "lga_code": ["A", "A", "B", "A", "B"],
            "state_name": ["Alpha"] * len(dates),
            "fatalities": [3, 1, 2, 4, 0],
            "actor_primary": ["Government of Nigeria"] * len(dates),
            "actor_secondary": ["Group X", "Group X", "Group Y", "Group X", None],
        }
    )
    return incidents, registry, adjacency


def test_label_marks_the_week_of_the_event(toy_inputs):
    incidents, registry, adjacency = toy_inputs
    result = build_panel(incidents, registry, adjacency)
    frame = result.frame

    positives = frame[frame["label"] == 1][["lga_code", "week_start"]]
    labelled = {(row.lga_code, row.week_start.date().isoformat()) for row in positives.itertuples()}
    assert ("A", "2013-03-04") in labelled
    assert ("A", "2013-03-11") in labelled
    assert ("B", "2016-06-06") in labelled
    assert ("A", "2019-09-02") in labelled
    # The state operation is not a positive label, it is a feature.
    assert ("B", "2022-04-04") not in labelled


def test_features_do_not_see_their_own_week(toy_inputs):
    incidents, registry, adjacency = toy_inputs
    frame = build_panel(incidents, registry, adjacency).frame

    first = frame[(frame["lga_code"] == "A") & (frame["week_start"] == pd.Timestamp("2013-03-04"))]
    assert first["own_events_1w"].iloc[0] == 0.0
    assert first["own_events_52w"].iloc[0] == 0.0

    second = frame[(frame["lga_code"] == "A") & (frame["week_start"] == pd.Timestamp("2013-03-11"))]
    assert second["own_events_1w"].iloc[0] == 1.0

    # B is adjacent to A, so it sees A's first event a week later but not sooner.
    neighbour_same_week = frame[
        (frame["lga_code"] == "B") & (frame["week_start"] == pd.Timestamp("2013-03-04"))
    ]
    assert neighbour_same_week["nb_events_1w"].iloc[0] == 0.0
    neighbour_next_week = frame[
        (frame["lga_code"] == "B") & (frame["week_start"] == pd.Timestamp("2013-03-11"))
    ]
    assert neighbour_next_week["nb_events_1w"].iloc[0] == 1.0


def test_panel_covers_every_lga_and_week(toy_inputs):
    incidents, registry, adjacency = toy_inputs
    frame = build_panel(incidents, registry, adjacency).frame
    assert set(frame["lga_code"]) == {"A", "B"}
    assert frame.groupby("lga_code").size().nunique() == 1
    assert not frame.duplicated(subset=["lga_code", "week_start"]).any()


def test_constant_features_are_held_out_of_the_model(toy_inputs):
    incidents, registry, adjacency = toy_inputs
    result = build_panel(incidents, registry, adjacency)
    # No documents or chatter were supplied, so those columns are constant zero.
    assert "chatter_volume_1w" in result.display_only
    assert "chatter_volume_1w" not in result.model_features
    assert "own_events_4w" in result.model_features
