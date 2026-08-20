"""UCDP Georeferenced Event Dataset.

UCDP is the backbone of the training corpus. Every event carries a start date, a
precision code for that date, latitude and longitude with their own precision
code, the two named parties, and separate death counts for each side and for
civilians. That combination is what makes a supervised weekly panel possible at
all: without a coordinate the event cannot be attributed to an LGA, and without a
date precision code there is no way to exclude events that are only known to the
month and would otherwise leak across week boundaries.

Two access routes are implemented. The versioned bulk CSV is public and needs no
credential, so it is the default and gives the full history back to 1989. The
REST API returns the same schema incrementally and is used when a token is
present, which is what a weekly production refresh would call.

Event classes are assigned from the UCDP violence typology. Non-state conflict
and one sided violence together are the closest available proxy for the banditry
and kidnapping problem the brief targets, since armed group activity against
civilians and between rival groups is coded there. State based violence is kept
but classed separately, because a military operation is a predictor of the next
week's risk rather than an instance of it.
"""

from __future__ import annotations

import csv
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import requests

from ..logging_utils import get_logger
from ..reference import assign_points_to_lgas, boundaries
from .base import Availability, Connector, Readiness

LOGGER = get_logger(__name__)

BULK_URL = "https://ucdp.uu.se/downloads/ged/ged{version}-csv.zip"
API_URL = "https://ucdpapi.pcr.uu.se/api/gedevents/{version}"
NIGERIA_COUNTRY_ID = 475
DEFAULT_VERSION = "251"

VIOLENCE_CLASS = {
    "1": "state_operation",       # state based armed conflict
    "2": "banditry_kidnapping",   # non-state conflict, includes armed group rivalry
    "3": "banditry_kidnapping",   # one sided violence against civilians
}
VIOLENCE_TYPE = {
    "1": "state_based_conflict",
    "2": "non_state_conflict",
    "3": "one_sided_violence",
}


class UcdpConnector(Connector):
    name = "ucdp"
    kind = "incident"
    cadence = "annual bulk, weekly candidate updates"
    description = "UCDP Georeferenced Event Dataset, georeferenced fatal events"

    def __init__(self, *args: Any, version: str = DEFAULT_VERSION, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.version = version
        self._archive = self.settings.paths.raw / f"ged{version}-csv.zip"

    # ------------------------------------------------------------ readiness
    def availability(self) -> Availability:
        if self._archive.exists():
            return Availability(Readiness.READY, "bulk CSV cached locally")
        try:
            response = requests.head(BULK_URL.format(version=self.version), timeout=20)
            if response.status_code < 400:
                return Availability(Readiness.READY, "bulk CSV reachable")
        except requests.RequestException as exc:
            return Availability(Readiness.NEEDS_NETWORK, str(exc.__class__.__name__))
        return Availability(Readiness.NEEDS_NETWORK, "bulk CSV endpoint returned an error")

    # -------------------------------------------------------------- fetching
    def _ensure_archive(self) -> Path:
        if self._archive.exists():
            return self._archive
        url = BULK_URL.format(version=self.version)
        LOGGER.info("downloading UCDP GED v%s", self.version)
        response = requests.get(url, timeout=900, stream=True)
        response.raise_for_status()
        self._archive.parent.mkdir(parents=True, exist_ok=True)
        with self._archive.open("wb") as handle:
            for block in response.iter_content(chunk_size=1 << 20):
                handle.write(block)
        return self._archive

    def _iter_bulk_rows(self) -> Iterator[dict[str, str]]:
        csv.field_size_limit(sys.maxsize)
        archive = self._ensure_archive()
        with zipfile.ZipFile(archive) as bundle:
            member = next(name for name in bundle.namelist() if name.lower().endswith(".csv"))
            with bundle.open(member) as raw:
                text = (line.decode("utf-8", errors="replace") for line in raw)
                for row in csv.DictReader(text):
                    if row.get("country") == "Nigeria":
                        yield row

    def _iter_api_rows(self, since: date, until: date) -> Iterator[dict[str, Any]]:
        token = self.settings.env("UCDP_API_TOKEN")
        page = 0
        while True:
            response = requests.get(
                API_URL.format(version=f"{self.version[:2]}.{self.version[2:]}"),
                params={
                    "pagesize": 1000,
                    "page": page,
                    "Country": NIGERIA_COUNTRY_ID,
                    "StartDate": since.isoformat(),
                    "EndDate": until.isoformat(),
                },
                headers={"Authorization": f"Bearer {token}"} if token else {},
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            yield from payload.get("Result", [])
            if page + 1 >= int(payload.get("TotalPages", 1)):
                return
            page += 1

    # ------------------------------------------------------------ normalise
    def fetch(self, since: date, until: date) -> list[dict[str, Any]]:
        import pandas as pd

        token = self.settings.env("UCDP_API_TOKEN")
        rows = list(self._iter_api_rows(since, until)) if token else list(self._iter_bulk_rows())
        if not rows:
            return []

        frame = pd.DataFrame(rows)
        frame["event_date"] = pd.to_datetime(frame["date_start"], errors="coerce").dt.date
        frame = frame[frame["event_date"].notna()]
        frame = frame[
            (frame["event_date"] >= since) & (frame["event_date"] <= until)
        ].reset_index(drop=True)
        if frame.empty:
            return []

        frame["lga_code"] = assign_points_to_lgas(frame, boundaries())
        registry = boundaries().frame.set_index("lga_code")["state_name"]

        records: list[dict[str, Any]] = []
        for row in frame.itertuples():
            violence = str(getattr(row, "type_of_violence", ""))
            lga_code = getattr(row, "lga_code", None)
            civilian = int(float(getattr(row, "deaths_civilians", 0) or 0))
            records.append(
                self._incident(
                    source_event_id=str(getattr(row, "id", "")),
                    event_date=row.event_date,
                    event_class=VIOLENCE_CLASS.get(violence, "other"),
                    event_type=VIOLENCE_TYPE.get(violence, "unknown"),
                    actor_primary=getattr(row, "side_a", None),
                    actor_secondary=getattr(row, "side_b", None),
                    dyad=getattr(row, "dyad_name", None),
                    lga_code=lga_code,
                    state_name=registry.get(lga_code) if lga_code else None,
                    latitude=float(getattr(row, "latitude", 0) or 0) or None,
                    longitude=float(getattr(row, "longitude", 0) or 0) or None,
                    geolocation_precision=_as_int(getattr(row, "where_prec", None)),
                    date_precision=_as_int(getattr(row, "date_prec", None)),
                    fatalities=int(float(getattr(row, "best", 0) or 0)),
                    civilian_deaths=civilian,
                    headline=getattr(row, "source_headline", None),
                    description=getattr(row, "where_description", None),
                    source_url=None,
                )
            )
        LOGGER.info("ucdp normalised %d Nigerian events", len(records))
        return records


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
