"""Common contract for the nine source producers.

Each source in the data ingestion strategy becomes one producer object. A
producer knows three things: whether it can run at all in the current
environment, how to pull a window of records from its source, and how to
normalise those records onto one of the three Avro contracts. Nothing else in the
pipeline knows which source a record came from beyond the ``source`` field, which
is what allows a source to be added or withdrawn without touching the feature
code.

Readiness is reported rather than assumed. A connector that needs a key it has
not been given reports ``needs_credentials`` and is skipped with a logged reason,
so a partial ingestion run is visible in the output instead of silently producing
an empty panel.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Sequence

import pandas as pd

from ..config import Settings, get_settings
from ..logging_utils import get_logger
from ..stream import EventBus

LOGGER = get_logger(__name__)


class Readiness(str, Enum):
    READY = "ready"
    NEEDS_CREDENTIALS = "needs_credentials"
    NEEDS_NETWORK = "needs_network"
    NOT_IMPLEMENTED = "not_implemented"


@dataclass
class Availability:
    state: Readiness
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.state is Readiness.READY


@dataclass
class IngestResult:
    producer: str
    kind: str
    records: int
    transport: str
    readiness: Readiness
    detail: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "kind": self.kind,
            "records": self.records,
            "transport": self.transport,
            "readiness": self.readiness.value,
            "detail": self.detail,
        }


def stable_id(*parts: Any) -> str:
    joined = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:24]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Connector(ABC):
    """One producer in the ingestion topology."""

    name: str = "connector"
    kind: str = "incident"          # incident, document or chatter
    cadence: str = "daily"
    description: str = ""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------ lifecycle
    @abstractmethod
    def availability(self) -> Availability:
        """Report whether this producer can run right now."""

    @abstractmethod
    def fetch(self, since: date, until: date) -> list[dict[str, Any]]:
        """Pull and normalise records for the window, inclusive of both bounds."""

    def run(
        self,
        bus: EventBus,
        since: date | None = None,
        until: date | None = None,
    ) -> IngestResult:
        availability = self.availability()
        if not availability.ready:
            LOGGER.info("skipping %s: %s (%s)", self.name, availability.state.value, availability.detail)
            return IngestResult(
                producer=self.name,
                kind=self.kind,
                records=0,
                transport=bus.transport,
                readiness=availability.state,
                detail=availability.detail,
            )

        until = until or date.today()
        since = since or (until - timedelta(days=30))
        records = self.fetch(since, until)
        published = bus.publish(self.kind, records, producer_name=self.name)
        return IngestResult(
            producer=self.name,
            kind=self.kind,
            records=published.count,
            transport=published.transport,
            readiness=Readiness.READY,
            detail=availability.detail,
        )

    # -------------------------------------------------------------- helpers
    def _incident(self, **fields: Any) -> dict[str, Any]:
        record = {
            "event_id": fields.get("event_id")
            or stable_id(self.name, fields.get("source_event_id"), fields.get("event_date")),
            "source": self.name,
            "source_event_id": fields.get("source_event_id"),
            "event_date": fields["event_date"],
            "event_class": fields.get("event_class", "other"),
            "event_type": fields.get("event_type"),
            "actor_primary": fields.get("actor_primary"),
            "actor_secondary": fields.get("actor_secondary"),
            "dyad": fields.get("dyad"),
            "lga_code": fields.get("lga_code"),
            "state_name": fields.get("state_name"),
            "latitude": fields.get("latitude"),
            "longitude": fields.get("longitude"),
            "geolocation_precision": fields.get("geolocation_precision"),
            "date_precision": fields.get("date_precision"),
            "fatalities": int(fields.get("fatalities") or 0),
            "civilian_deaths": int(fields.get("civilian_deaths") or 0),
            "victims": fields.get("victims"),
            "ransom_ngn": fields.get("ransom_ngn"),
            "headline": fields.get("headline"),
            "description": fields.get("description"),
            "source_url": fields.get("source_url"),
            "ingested_at": utc_now(),
        }
        return record

    def _document(self, **fields: Any) -> dict[str, Any]:
        return {
            "doc_id": fields.get("doc_id") or stable_id(self.name, fields.get("url"), fields.get("title")),
            "source": self.name,
            "doc_type": fields.get("doc_type"),
            "title": fields.get("title"),
            "published_at": fields.get("published_at"),
            "url": fields.get("url"),
            "body": fields.get("body") or "",
            "lga_codes": list(fields.get("lga_codes") or []),
            "ransom_ngn": fields.get("ransom_ngn"),
            "ingested_at": utc_now(),
        }

    def _chatter(self, **fields: Any) -> dict[str, Any]:
        return {
            "chatter_id": fields.get("chatter_id") or stable_id(self.name, fields.get("url"), fields.get("body")),
            "platform": fields.get("platform", self.name),
            "posted_at": fields["posted_at"],
            "lga_code": fields.get("lga_code"),
            "body": fields.get("body") or "",
            "sentiment": fields.get("sentiment"),
            "threat_score": fields.get("threat_score"),
            "url": fields.get("url"),
            "ingested_at": utc_now(),
        }


def within_window(values: Sequence[Any], since: date, until: date) -> pd.Series:
    stamps = pd.to_datetime(pd.Series(list(values)), errors="coerce")
    return (stamps >= pd.Timestamp(since)) & (stamps <= pd.Timestamp(until))
