"""Run the nine producers, then drain the topics into the warehouse.

The topology has two halves that are deliberately kept apart. Producers know how
to talk to a publisher and how to fill in an Avro contract, and they stop there.
A single consumer reads the topics back and is the only component that writes to
the warehouse. Splitting it this way means a source can fail, be rate limited or
lack a key without leaving the warehouse half written, and a replay of a topic
reconstructs the tables exactly because every write is an upsert on a natural key.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Iterable, Sequence

import pandas as pd

from ..config import Settings, get_settings
from ..logging_utils import get_logger
from ..storage import init_schema, upsert
from ..stream import EventBus
from .acled import AcledConnector
from .base import Connector, IngestResult
from .documents import (
    HumAngleConnector,
    HumanitarianConnector,
    NbsConnector,
    NextierConnector,
    PressConnector,
    SbmConnector,
)
from .social import SocialConnector
from .ucdp import UcdpConnector

LOGGER = get_logger(__name__)

PRODUCERS: tuple[type[Connector], ...] = (
    AcledConnector,
    UcdpConnector,
    SbmConnector,
    NextierConnector,
    HumAngleConnector,
    SocialConnector,
    NbsConnector,
    PressConnector,
    HumanitarianConnector,
)


def build_producers(settings: Settings | None = None) -> list[Connector]:
    settings = settings or get_settings()
    return [producer(settings) for producer in PRODUCERS]


def run_ingestion(
    since: date | None = None,
    until: date | None = None,
    only: Sequence[str] | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Execute the producers and return one row per producer describing the outcome."""
    settings = settings or get_settings()
    until = until or date.today()
    since = since or (until - timedelta(days=30))
    bus = EventBus(settings)

    results: list[IngestResult] = []
    try:
        for connector in build_producers(settings):
            if only and connector.name not in only:
                continue
            try:
                results.append(connector.run(bus, since=since, until=until))
            except Exception as exc:  # noqa: BLE001 - one failing publisher must not stop the run
                LOGGER.error("producer %s failed: %s", connector.name, exc)
                results.append(
                    IngestResult(
                        producer=connector.name,
                        kind=connector.kind,
                        records=0,
                        transport=bus.transport,
                        readiness=connector.availability().state,
                        detail=f"{exc.__class__.__name__}: {exc}",
                    )
                )
    finally:
        bus.close()

    summary = pd.DataFrame([result.as_row() for result in results])
    if not summary.empty:
        LOGGER.info(
            "ingestion complete: %d records from %d of %d producers",
            int(summary["records"].sum()),
            int((summary["records"] > 0).sum()),
            len(summary),
        )
    return summary


def drain_to_warehouse(settings: Settings | None = None) -> dict[str, int]:
    """Consume every topic and upsert the records into their warehouse tables."""
    settings = settings or get_settings()
    init_schema(settings)
    bus = EventBus(settings)
    written: dict[str, int] = {}
    try:
        written["incidents"] = _write_incidents(bus.replay("incident"))
        written["documents"] = _write_documents(bus.replay("document"))
        written["chatter"] = _write_chatter(bus.replay("chatter"))
    finally:
        bus.close()
    LOGGER.info("warehouse load: %s", json.dumps(written))
    return written


def _batched(records: Iterable[dict], size: int = 2000) -> Iterable[list[dict]]:
    batch: list[dict] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_incidents(records: Iterable[dict]) -> int:
    total = 0
    for batch in _batched(records):
        rows = []
        for record in batch:
            row = dict(record)
            row["event_date"] = _as_iso(row.get("event_date"))
            row["geom_wkt"] = (
                f"POINT ({row['longitude']} {row['latitude']})"
                if row.get("longitude") is not None and row.get("latitude") is not None
                else None
            )
            rows.append(row)
        total += upsert("incidents", _project(rows, _INCIDENT_COLUMNS))
    return total


def _write_documents(records: Iterable[dict]) -> int:
    total = 0
    for batch in _batched(records):
        rows = []
        for record in batch:
            row = dict(record)
            row["published_at"] = _as_iso(row.get("published_at"))
            row["lga_codes"] = json.dumps(list(row.get("lga_codes") or []))
            rows.append(row)
        total += upsert("documents", _project(rows, _DOCUMENT_COLUMNS))
    return total


def _write_chatter(records: Iterable[dict]) -> int:
    total = 0
    for batch in _batched(records):
        total += upsert("chatter", _project([dict(row) for row in batch], _CHATTER_COLUMNS))
    return total


_INCIDENT_COLUMNS = (
    "event_id", "source", "source_event_id", "event_date", "event_class", "event_type",
    "actor_primary", "actor_secondary", "dyad", "lga_code", "state_name", "latitude",
    "longitude", "geolocation_precision", "date_precision", "fatalities",
    "civilian_deaths", "victims", "ransom_ngn", "headline", "description",
    "source_url", "ingested_at", "geom_wkt",
)
_DOCUMENT_COLUMNS = (
    "doc_id", "source", "doc_type", "title", "published_at", "url", "body",
    "lga_codes", "ransom_ngn", "ingested_at",
)
_CHATTER_COLUMNS = (
    "chatter_id", "platform", "posted_at", "lga_code", "body", "sentiment",
    "threat_score", "url", "ingested_at",
)


def _project(rows: list[dict], columns: tuple[str, ...]) -> list[dict]:
    from ..storage.engine import get_engine

    _, backend = get_engine()
    keep = list(columns)
    if backend.is_postgres and backend.postgis:
        keep = [column if column != "geom_wkt" else "geom" for column in keep]
    if not backend.is_postgres:
        keep = [column for column in keep if column != "geom"]
    projected = []
    for row in rows:
        projected.append({column: row.get(column if column != "geom" else "geom_wkt") for column in keep})
    return projected


def _as_iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value[:10]
    if isinstance(value, (int, float)):
        # Avro date logical type is days since the Unix epoch.
        return (date(1970, 1, 1) + timedelta(days=int(value))).isoformat()
    return value.isoformat()[:10]
