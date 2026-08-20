"""The unstructured producers: SBM, Nextier, HumAngle, press and humanitarian.

These five sources share a shape. Something publishes an index, whether an RSS
feed, a category listing or a dataset landing page; each entry resolves to an
HTML article or a PDF report; and the useful content has to be pulled out of that
document with entity extraction rather than read from a field. They therefore
share one base class and differ only in how they enumerate candidate documents
and how they label what they find.

The extraction step attaches LGA codes, ransom figures and casualty counts to
each document at ingestion time rather than at feature time. Doing it once at the
boundary means a document is stored with its resolved entities, so the feature
pipeline joins on codes and never re-parses free text, and a mis-resolved place
name can be corrected in one table instead of being recomputed everywhere.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import date, timedelta
from typing import Any, Iterable

from ..logging_utils import get_logger
from ..nlp.extract import Gazetteer, extract
from ..reference import boundaries
from .base import Availability, Connector, Readiness, stable_id
from .web import Fetcher, html_links, html_to_text, parse_date, pdf_to_text

LOGGER = get_logger(__name__)


class DocumentConnector(Connector):
    """Enumerate documents from an index, extract entities, emit Document records."""

    kind = "document"
    index_urls: tuple[str, ...] = ()
    doc_type = "report"
    max_documents = 40

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fetcher = Fetcher(self.settings)
        self._gazetteer: Gazetteer | None = None
        self._summaries: dict[str, str] = {}

    @property
    def gazetteer(self) -> Gazetteer:
        if self._gazetteer is None:
            self._gazetteer = Gazetteer.from_registry(boundaries().frame)
        return self._gazetteer

    def availability(self) -> Availability:
        if not self.index_urls:
            return Availability(Readiness.NOT_IMPLEMENTED, "no index configured")
        for url in self.index_urls:
            if self.fetcher.reachable(url):
                return Availability(Readiness.READY, f"index reachable at {url}")
        return Availability(
            Readiness.NEEDS_NETWORK,
            "publisher index is not reachable from this environment",
        )

    @abstractmethod
    def discover(self, since: date, until: date) -> Iterable[tuple[str, str, date | None]]:
        """Yield candidate documents as url, title, published date."""

    def fetch(self, since: date, until: date) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for url, title, published in list(self.discover(since, until))[: self.max_documents]:
            payload = self.fetcher.get(url)
            # Many Nigerian news sites render the article body client side, so the
            # fetched page yields little text. The feed summary is authored server
            # side and usually carries the lede, which is where the place names
            # are, so both are kept and extraction runs over the combination.
            summary = self._summaries.get(url, "")
            fetched = ""
            if payload:
                fetched = (
                    pdf_to_text(payload)
                    if url.lower().endswith(".pdf")
                    else html_to_text(payload)
                )
            body = " ".join(part for part in (summary, fetched) if part).strip()
            if len(body) < 120:
                continue
            facts = extract(f"{title}. {body}", self.gazetteer)
            records.append(
                self._document(
                    doc_id=stable_id(self.name, url),
                    doc_type=self.doc_type,
                    title=title,
                    published_at=published or date.today(),
                    url=url,
                    body=body[:20000],
                    lga_codes=facts.lga_codes,
                    ransom_ngn=facts.ransom_ngn,
                )
            )
        LOGGER.info("%s produced %d documents", self.name, len(records))
        return records

    # -------------------------------------------------------------- helpers
    def _links_from_index(
        self, url: str, keywords: tuple[str, ...], suffix: str | None = None
    ) -> list[tuple[str, str, date | None]]:
        payload = self.fetcher.get(url)
        if not payload:
            return []
        found: list[tuple[str, str, date | None]] = []
        seen: set[str] = set()
        for link, title in html_links(payload, url):
            lowered = f"{link} {title}".lower()
            if suffix and not link.lower().endswith(suffix):
                continue
            if keywords and not any(keyword in lowered for keyword in keywords):
                continue
            if link in seen:
                continue
            seen.add(link)
            found.append((link, title, None))
        return found

    def _links_from_feed(self, feed_url: str) -> list[tuple[str, str, date | None]]:
        import feedparser

        payload = self.fetcher.get(feed_url)
        if not payload:
            return []
        parsed = feedparser.parse(payload)
        entries: list[tuple[str, str, date | None]] = []
        for entry in parsed.entries:
            link = entry.get("link")
            if not link:
                continue
            published = parse_date(entry.get("published") or entry.get("updated"))
            summary = entry.get("summary") or ""
            if not summary and entry.get("content"):
                summary = entry["content"][0].get("value", "")
            if summary:
                self._summaries[link] = html_to_text(summary.encode("utf-8"))
            entries.append((link, entry.get("title", ""), published))
        return entries


class SbmConnector(DocumentConnector):
    """SBM Intelligence analyst reports and security advisories.

    SBM is the source the brief cites for the concentration of victims in the
    North West and for state level abduction totals. Its value to the model is
    not the incident count, which ACLED and UCDP already carry, but the ransom
    economics, which no structured feed publishes and which the extraction layer
    pulls out of the report text.
    """

    name = "sbm"
    doc_type = "analyst_report"
    index_urls = ("https://www.sbmintel.com/reports/", "https://www.sbmintel.com/blog/")

    def discover(self, since: date, until: date) -> Iterable[tuple[str, str, date | None]]:
        for url in self.index_urls:
            yield from self._links_from_index(
                url, keywords=("kidnap", "security", "banditry", "ransom", "report", "insecurity")
            )


class NextierConnector(DocumentConnector):
    """Nextier Nigeria Violent Conflict database commentary and weekly updates."""

    name = "nextier"
    doc_type = "conflict_update"
    index_urls = (
        "https://nextierspd.com/nigeria-violent-conflict-database/",
        "https://nextierspd.com/category/violent-conflict/",
    )

    def discover(self, since: date, until: date) -> Iterable[tuple[str, str, date | None]]:
        for url in self.index_urls:
            yield from self._links_from_index(
                url, keywords=("conflict", "violence", "kidnap", "attack", "weekly", "update")
            )


class HumAngleConnector(DocumentConnector):
    """HumAngle field reporting and its abduction tracker.

    HumAngle publishes no API, so the feed is the entry point and the article
    body is parsed. Its reporting is often the earliest published account of a
    rural incident, which is why it is retained despite the extraction cost.
    """

    name = "humangle"
    doc_type = "field_report"
    index_urls = ("https://humanglemedia.com/feed/", "https://humanglemedia.com/category/security/")

    def discover(self, since: date, until: date) -> Iterable[tuple[str, str, date | None]]:
        entries = self._links_from_feed(self.index_urls[0])
        if entries:
            yield from entries
            return
        yield from self._links_from_index(
            self.index_urls[1], keywords=("kidnap", "abduct", "bandit", "attack", "security")
        )


class PressConnector(DocumentConnector):
    """Nigeria Police Force releases and national newspaper security desks.

    Press reporting is treated as a corroboration and timing signal rather than a
    count. A newspaper report rarely adds an incident the coded feeds miss
    entirely, but it usually arrives days earlier, and the volume of reporting
    naming an LGA is itself informative about attention and about pressure on the
    security response.
    """

    name = "press"
    doc_type = "news_article"
    index_urls = (
        "https://dailytrust.com/feed/",
        "https://www.vanguardngr.com/feed/",
        "https://www.npf.gov.ng/feed/",
    )
    max_documents = 60

    def discover(self, since: date, until: date) -> Iterable[tuple[str, str, date | None]]:
        for url in self.index_urls:
            for link, title, published in self._links_from_feed(url):
                if published and not (since <= published <= until):
                    continue
                yield link, title, published


class HumanitarianConnector(DocumentConnector):
    """IOM displacement tracking and UN OCHA situation reports on HDX.

    Displacement is a consequence of violence rather than a precursor, so these
    records are used as a slow moving indicator of sustained pressure on an area:
    an LGA that has been absorbing displacement for several weeks is under
    conditions that sustain armed group presence.
    """

    name = "humanitarian"
    doc_type = "situation_report"
    index_urls = (
        "https://data.humdata.org/api/3/action/package_search?q=nigeria+displacement&rows=25",
        "https://reliefweb.int/updates/rss.xml?advanced-search=%28C182%29",
    )

    def discover(self, since: date, until: date) -> Iterable[tuple[str, str, date | None]]:
        import json

        payload = self.fetcher.get(self.index_urls[0])
        if payload:
            try:
                catalogue = json.loads(payload)
                for dataset in catalogue.get("result", {}).get("results", []):
                    for resource in dataset.get("resources", []):
                        if str(resource.get("format", "")).upper() in {"PDF", "CSV", "XLSX"}:
                            yield (
                                resource["url"],
                                f"{dataset.get('title', '')}: {resource.get('name', '')}",
                                parse_date(str(dataset.get("metadata_modified", ""))[:10]),
                            )
            except (ValueError, KeyError):
                pass
        yield from self._links_from_feed(self.index_urls[1])


class NbsConnector(DocumentConnector):
    """National Bureau of Statistics crime and security statistics releases.

    NBS publishes state level rather than LGA level counts and arrives with a
    long lag, so it cannot drive a weekly score. It enters the model as a slowly
    varying state level prior that stabilises predictions for LGAs with sparse
    event histories, which is precisely where a purely event driven model is
    least reliable.
    """

    name = "nbs"
    doc_type = "official_statistics"
    cadence = "annual"
    index_urls = ("https://nigerianstat.gov.ng/elibrary?queries[search]=crime",)

    def discover(self, since: date, until: date) -> Iterable[tuple[str, str, date | None]]:
        yield from self._links_from_index(
            self.index_urls[0], keywords=("crime", "security", "insecurity")
        )


DOCUMENT_CONNECTORS = (
    SbmConnector,
    NextierConnector,
    HumAngleConnector,
    PressConnector,
    HumanitarianConnector,
    NbsConnector,
)
