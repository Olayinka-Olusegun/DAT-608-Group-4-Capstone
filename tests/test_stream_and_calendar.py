"""Transport contracts and calendar arithmetic.

The Avro layer is the boundary every producer crosses, so the round trip and the
compatibility rules are tested directly rather than through a connector. The
calendar is tested against dates that can be checked by hand, because a festival
flag that is a week out would be worse than no flag at all.
"""

from __future__ import annotations

from datetime import date

import pytest

from pau_risk.reference.calendar_ng import (
    easter_sunday,
    festival_within,
    holidays_for_year,
    is_dry_season,
)
from pau_risk.stream.registry import (
    IncompatibleSchemaError,
    SchemaRegistry,
    check_backward_compatible,
    load_schema_file,
)


# ------------------------------------------------------------------- schemas
def test_every_declared_schema_parses():
    for name in ("incident", "document", "chatter"):
        schema = load_schema_file(name)
        assert schema["type"] == "record"
        assert schema["fields"]


def test_avro_round_trip_preserves_a_record(tmp_path, monkeypatch):
    from pau_risk.config import get_settings
    from pau_risk.stream.bus import EventBus

    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:1")  # force the file sink
    settings = get_settings()
    bus = EventBus(settings)
    record = {
        "event_id": "abc123",
        "source": "test",
        "event_date": "2024-05-06",
        "event_class": "banditry_kidnapping",
        "fatalities": 4,
        "civilian_deaths": 2,
        "latitude": 11.5,
        "longitude": 7.25,
        "ingested_at": "2024-05-07T00:00:00+00:00",
    }
    decoded = bus.decode("incident", bus.encode("incident", record))
    assert decoded["event_id"] == "abc123"
    assert decoded["event_date"] == date(2024, 5, 6)
    assert decoded["fatalities"] == 4
    assert decoded["actor_primary"] is None  # default applied
    bus.close()


def test_adding_a_field_with_a_default_is_compatible():
    previous = {"type": "record", "name": "R", "fields": [{"name": "a", "type": "string"}]}
    candidate = {
        "type": "record",
        "name": "R",
        "fields": [
            {"name": "a", "type": "string"},
            {"name": "b", "type": ["null", "string"], "default": None},
        ],
    }
    check_backward_compatible(previous, candidate)


def test_removing_a_field_is_rejected():
    previous = {
        "type": "record", "name": "R",
        "fields": [{"name": "a", "type": "string"}, {"name": "b", "type": "int"}],
    }
    candidate = {"type": "record", "name": "R", "fields": [{"name": "a", "type": "string"}]}
    with pytest.raises(IncompatibleSchemaError, match="fields removed"):
        check_backward_compatible(previous, candidate)


def test_adding_a_field_without_a_default_is_rejected():
    previous = {"type": "record", "name": "R", "fields": [{"name": "a", "type": "string"}]}
    candidate = {
        "type": "record", "name": "R",
        "fields": [{"name": "a", "type": "string"}, {"name": "b", "type": "int"}],
    }
    with pytest.raises(IncompatibleSchemaError, match="no default"):
        check_backward_compatible(previous, candidate)


def test_registry_is_idempotent_and_versions(tmp_path, monkeypatch):
    from pau_risk.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(
        type(settings), "schema_registry_path",
        property(lambda self: tmp_path / "registry.json"),
    )
    registry = SchemaRegistry(settings)
    schema = load_schema_file("chatter")
    first = registry.register("nga.chatter.v1-value", schema)
    second = registry.register("nga.chatter.v1-value", schema)
    assert first.version == second.version == 1
    assert first.schema_id == second.schema_id

    extended = dict(schema)
    extended["fields"] = schema["fields"] + [
        {"name": "language", "type": ["null", "string"], "default": None}
    ]
    third = registry.register("nga.chatter.v1-value", extended)
    assert third.version == 2


# ------------------------------------------------------------------ calendar
@pytest.mark.parametrize(
    ("year", "expected"),
    [(2023, date(2023, 4, 9)), (2024, date(2024, 3, 31)), (2025, date(2025, 4, 20))],
)
def test_easter_computus(year, expected):
    assert easter_sunday(year) == expected


def test_fixed_national_holidays_are_present():
    holidays = holidays_for_year(2024)
    assert holidays[date(2024, 10, 1)] == "Independence Day"
    assert holidays[date(2024, 6, 12)] == "Democracy Day"
    assert holidays[date(2024, 12, 25)] == "Christmas"


def test_democracy_day_moved_in_2019():
    assert date(2018, 5, 29) in holidays_for_year(2018)
    assert date(2019, 6, 12) in holidays_for_year(2019)


def test_festival_window_covers_the_horizon_only():
    flagged, name = festival_within(date(2024, 9, 26), days=7)
    assert flagged == 1 and name == "Independence Day"
    quiet, _ = festival_within(date(2024, 9, 1), days=7)
    assert quiet == 0


def test_dry_season_covers_november_to_march():
    assert is_dry_season(date(2024, 1, 15)) == 1
    assert is_dry_season(date(2024, 12, 15)) == 1
    assert is_dry_season(date(2024, 7, 15)) == 0
