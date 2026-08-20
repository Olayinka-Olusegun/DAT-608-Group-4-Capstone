"""ACLED, the highest cadence structured feed in the strategy.

ACLED refreshes weekly and codes at a finer event granularity than UCDP,
separating abduction and forced disappearance from armed clash and from violence
against civilians. That separation matters for this model because the target is
specifically kidnapping and banditry rather than conflict fatalities, and ACLED
is the only feed that labels an abduction with no deaths, which UCDP by
construction never records.

The connector is written against the current ACLED read API, which authenticates
with an access key and the registered account email. Without a key the connector
reports ``needs_credentials`` and the run proceeds on the remaining sources, so
the pipeline degrades to a smaller event base rather than failing.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..logging_utils import get_logger
from ..reference import assign_points_to_lgas, boundaries
from .base import Availability, Connector, Readiness

LOGGER = get_logger(__name__)

ACLED_URL = "https://api.acleddata.com/acled/read"
PAGE_SIZE = 5000

# ACLED sub event types mapped onto the two classes the model separates.
BANDITRY_SUB_EVENTS = {
    "abduction/forced disappearance",
    "attack",
    "armed clash",
    "looting/property destruction",
    "sexual violence",
    "mob violence",
}
STATE_OPERATION_SUB_EVENTS = {
    "air/drone strike",
    "shelling/artillery/missile attack",
    "government regains territory",
    "non-state actor overtakes territory",
    "disrupted weapons use",
}


class AcledConnector(Connector):
    name = "acled"
    kind = "incident"
    cadence = "weekly"
    description = "ACLED read API, coded political violence events"

    def availability(self) -> Availability:
        key = self.settings.env("ACLED_API_KEY")
        email = self.settings.env("ACLED_EMAIL")
        if not key or not email:
            return Availability(
                Readiness.NEEDS_CREDENTIALS,
                "set ACLED_API_KEY and ACLED_EMAIL, requested at developer.acleddata.com",
            )
        return Availability(Readiness.READY, "API key present")

    def _session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def fetch(self, since: date, until: date) -> list[dict[str, Any]]:
        import pandas as pd

        session = self._session()
        page, collected = 1, []
        while True:
            response = session.get(
                ACLED_URL,
                params={
                    "key": self.settings.env("ACLED_API_KEY"),
                    "email": self.settings.env("ACLED_EMAIL"),
                    "country": "Nigeria",
                    "event_date": f"{since.isoformat()}|{until.isoformat()}",
                    "event_date_where": "BETWEEN",
                    "limit": PAGE_SIZE,
                    "page": page,
                },
                timeout=180,
            )
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("data", [])
            collected.extend(batch)
            if len(batch) < PAGE_SIZE:
                break
            page += 1

        if not collected:
            return []

        frame = pd.DataFrame(collected)
        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce").dt.date
        frame = frame[frame["event_date"].notna()].reset_index(drop=True)
        frame["lga_code"] = assign_points_to_lgas(frame, boundaries())

        records = []
        for row in frame.itertuples():
            sub_event = str(getattr(row, "sub_event_type", "")).strip().lower()
            if sub_event in STATE_OPERATION_SUB_EVENTS or "military forces of nigeria" in str(
                getattr(row, "actor1", "")
            ).lower():
                event_class = "state_operation"
            elif sub_event in BANDITRY_SUB_EVENTS:
                event_class = "banditry_kidnapping"
            else:
                event_class = "other"
            records.append(
                self._incident(
                    source_event_id=str(getattr(row, "event_id_cnty", "")),
                    event_date=row.event_date,
                    event_class=event_class,
                    event_type=sub_event or None,
                    actor_primary=getattr(row, "actor1", None),
                    actor_secondary=getattr(row, "actor2", None),
                    dyad=None,
                    lga_code=getattr(row, "lga_code", None),
                    state_name=getattr(row, "admin1", None),
                    latitude=_as_float(getattr(row, "latitude", None)),
                    longitude=_as_float(getattr(row, "longitude", None)),
                    geolocation_precision=_as_int(getattr(row, "geo_precision", None)),
                    date_precision=None,
                    fatalities=_as_int(getattr(row, "fatalities", 0)) or 0,
                    headline=getattr(row, "notes", None),
                    description=getattr(row, "location", None),
                    source_url=getattr(row, "source", None),
                )
            )
        LOGGER.info("acled normalised %d events", len(records))
        return records


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
