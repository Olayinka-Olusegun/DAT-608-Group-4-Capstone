"""Typed read and write helpers over the warehouse.

Upserts are expressed once and dispatched to the dialect-specific conflict
clause, which keeps every ingestion connector idempotent: replaying a Kafka
partition or re-running a scraper cannot duplicate rows.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any, Iterable, Sequence

import pandas as pd
from sqlalchemy import text

from ..logging_utils import get_logger
from .engine import get_engine, read_sql

LOGGER = get_logger(__name__)

_KEYS: dict[str, tuple[str, ...]] = {
    "lga_registry": ("lga_code",),
    "lga_adjacency": ("lga_code", "neighbour_code"),
    "incidents": ("event_id",),
    "documents": ("doc_id",),
    "chatter": ("chatter_id",),
    "lga_week_features": ("lga_code", "week_start"),
    "model_runs": ("run_id",),
    "predictions": ("run_id", "lga_code", "week_start"),
    "prediction_drivers": ("run_id", "lga_code", "week_start", "driver_rank"),
    "threat_actor_edges": ("run_id", "actor", "target_kind", "target"),
    "security_briefs": ("brief_id",),
    "alerts": ("alert_id",),
}

_JSON_COLUMNS = {"metrics", "features", "payload", "lga_codes", "embedding"}


def _coerce(value: Any, column: str, is_postgres: bool) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if column in _JSON_COLUMNS and not isinstance(value, str):
        if is_postgres and column == "lga_codes":
            return list(value)
        return json.dumps(value, default=str)
    if isinstance(value, (datetime, date)):
        return value if is_postgres else value.isoformat()
    if isinstance(value, (pd.Timestamp,)):
        return value.to_pydatetime() if is_postgres else value.isoformat()
    return value


def upsert(table: str, rows: Sequence[dict[str, Any]], chunk_size: int = 2000) -> int:
    """Insert rows, replacing any that collide on the table's natural key."""
    rows = [row for row in rows if row]
    if not rows:
        return 0
    if table not in _KEYS:
        raise KeyError(f"No natural key registered for table {table}")

    engine, backend = get_engine()
    columns = list(rows[0].keys())
    key = [column for column in _KEYS[table] if column in columns]
    updatable = [column for column in columns if column not in key]

    placeholders = ", ".join(f":{column}" for column in columns)
    column_list = ", ".join(columns)
    if backend.is_postgres and updatable:
        assignments = ", ".join(f"{column} = EXCLUDED.{column}" for column in updatable)
        statement = (
            f"INSERT INTO {table} ({column_list}) VALUES ({placeholders}) "
            f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {assignments}"
        )
    elif backend.is_postgres:
        statement = (
            f"INSERT INTO {table} ({column_list}) VALUES ({placeholders}) "
            f"ON CONFLICT ({', '.join(key)}) DO NOTHING"
        )
    else:
        statement = (
            f"INSERT OR REPLACE INTO {table} ({column_list}) VALUES ({placeholders})"
        )

    payload = [
        {column: _coerce(row.get(column), column, backend.is_postgres) for column in columns}
        for row in rows
    ]
    written = 0
    with engine.begin() as connection:
        for start in range(0, len(payload), chunk_size):
            batch = payload[start : start + chunk_size]
            connection.execute(text(statement), batch)
            written += len(batch)
    return written


def upsert_frame(table: str, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return upsert(table, frame.to_dict(orient="records"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ readers
def load_registry() -> pd.DataFrame:
    return read_sql(
        """
        SELECT lga_code, lga_name, state_code, state_name, zone,
               senatorial_district, area_sqkm, centre_lat, centre_lon
        FROM lga_registry
        ORDER BY lga_code
        """
    )


def load_adjacency() -> pd.DataFrame:
    return read_sql(
        "SELECT lga_code, neighbour_code, weight, centroid_km FROM lga_adjacency"
    )


def load_incidents(classes: Iterable[str] | None = None) -> pd.DataFrame:
    query = """
        SELECT event_id, source, event_date, event_class, event_type,
               actor_primary, actor_secondary, dyad, lga_code, state_name,
               latitude, longitude, fatalities, civilian_deaths, victims,
               ransom_ngn, headline
        FROM incidents
        WHERE lga_code IS NOT NULL
    """
    frame = read_sql(query)
    if classes is not None:
        frame = frame[frame["event_class"].isin(list(classes))]
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    return frame.reset_index(drop=True)


def latest_run_id() -> str | None:
    frame = read_sql(
        "SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1"
    )
    return None if frame.empty else str(frame.iloc[0]["run_id"])


def load_predictions(run_id: str | None = None, limit: int | None = None) -> pd.DataFrame:
    run_id = run_id or latest_run_id()
    if run_id is None:
        return pd.DataFrame()
    query = """
        SELECT p.run_id, p.lga_code, r.lga_name, r.state_name, r.zone,
               r.centre_lat, r.centre_lon, p.week_start, p.probability,
               p.risk_tier, p.rank_national, p.rank_state
        FROM predictions p
        JOIN lga_registry r ON r.lga_code = p.lga_code
        WHERE p.run_id = :run_id
        ORDER BY p.probability DESC
    """
    frame = read_sql(query, {"run_id": run_id})
    return frame.head(limit) if limit else frame


def load_drivers(run_id: str, lga_code: str | None = None) -> pd.DataFrame:
    query = """
        SELECT run_id, lga_code, week_start, driver_rank, feature_name,
               feature_label, feature_value, shap_value
        FROM prediction_drivers
        WHERE run_id = :run_id
    """
    params: dict[str, Any] = {"run_id": run_id}
    if lga_code:
        query += " AND lga_code = :lga_code"
        params["lga_code"] = lga_code
    query += " ORDER BY lga_code, driver_rank"
    return read_sql(query, params)


def table_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in _KEYS:
        try:
            frame = read_sql(f"SELECT COUNT(*) AS n FROM {table}")
            counts[table] = int(frame.iloc[0]["n"])
        except Exception:  # noqa: BLE001 - table may not exist yet
            counts[table] = 0
    return counts
