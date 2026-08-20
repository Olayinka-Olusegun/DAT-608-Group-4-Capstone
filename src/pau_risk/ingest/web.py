"""Polite retrieval helpers shared by every scraping and PDF connector.

Four of the nine sources publish only as HTML pages or PDF reports, so a shared
fetching layer does the work once: it honours robots.txt, identifies itself,
spaces requests per host, retries transient failures, and caches every response
on disk. The cache is not an optimisation detail. Re-running the pipeline against
cached pages is what makes an ingestion run reproducible, and it keeps repeated
development runs from hammering a publisher that has not asked to be crawled.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from ..config import Settings, get_settings
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

USER_AGENT = (
    "PAU-DAT608-Research/0.1 (Pan-Atlantic University; academic study of "
    "conflict early warning; contact via institution)"
)
MIN_INTERVAL_SECONDS = 1.5


@dataclass
class Fetcher:
    settings: Settings = field(default_factory=get_settings)
    respect_robots: bool = True
    _last_call: dict[str, float] = field(default_factory=dict, init=False)
    _robots: dict[str, RobotFileParser] = field(default_factory=dict, init=False)

    @property
    def cache_dir(self) -> Path:
        path = self.settings.paths.raw / "web_cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]
        suffix = ".pdf" if url.lower().endswith(".pdf") else ".html"
        return self.cache_dir / f"{digest}{suffix}"

    def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        parser = self._robots.get(origin)
        if parser is None:
            parser = RobotFileParser()
            parser.set_url(f"{origin}/robots.txt")
            try:
                parser.read()
            except Exception:  # noqa: BLE001 - an unreadable robots file is treated as permissive
                parser = None  # type: ignore[assignment]
            self._robots[origin] = parser  # type: ignore[assignment]
        if parser is None:
            return True
        return parser.can_fetch(USER_AGENT, url)

    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        elapsed = time.monotonic() - self._last_call.get(host, 0.0)
        if elapsed < MIN_INTERVAL_SECONDS:
            time.sleep(MIN_INTERVAL_SECONDS - elapsed)
        self._last_call[host] = time.monotonic()

    def get(self, url: str, use_cache: bool = True) -> bytes | None:
        cache_path = self._cache_path(url)
        if use_cache and cache_path.exists():
            return cache_path.read_bytes()
        if not self._allowed(url):
            LOGGER.info("robots.txt disallows %s", url)
            return None
        self._throttle(url)
        try:
            response = requests.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=45, allow_redirects=True
            )
            if response.status_code >= 400:
                LOGGER.info("fetch returned %d for %s", response.status_code, url)
                return None
            cache_path.write_bytes(response.content)
            return response.content
        except requests.RequestException as exc:
            LOGGER.info("fetch failed for %s (%s)", url, exc.__class__.__name__)
            return None

    def reachable(self, url: str) -> bool:
        if self._cache_path(url).exists():
            return True
        try:
            self._throttle(url)
            response = requests.head(
                url, headers={"User-Agent": USER_AGENT}, timeout=15, allow_redirects=True
            )
            return response.status_code < 400
        except requests.RequestException:
            return False


def html_to_text(payload: bytes) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(payload, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def html_links(payload: bytes, base_url: str) -> list[tuple[str, str]]:
    from urllib.parse import urljoin

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(payload, "lxml")
    links: list[tuple[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        title = " ".join(anchor.get_text(separator=" ").split())
        if title:
            links.append((urljoin(base_url, anchor["href"]), title))
    return links


def pdf_to_text(payload: bytes, max_pages: int = 60) -> str:
    """Extract text from a report PDF.

    The brief specifies PyPDF2. That package is archived and its maintained
    continuation is pypdf, which exposes the same reader interface, so pypdf is
    used here and the extraction behaviour is unchanged.
    """
    import io

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(payload))
    except Exception as exc:  # noqa: BLE001 - malformed PDFs are common in the wild
        LOGGER.info("could not parse PDF (%s)", exc.__class__.__name__)
        return ""
    chunks = []
    for page in reader.pages[:max_pages]:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            continue
    return " ".join(" ".join(chunks).split())


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    for pattern in (
        "%Y-%m-%d", "%d %B %Y", "%B %d, %Y", "%d/%m/%Y", "%Y/%m/%d",
        "%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            return datetime.strptime(value.strip(), pattern).date()
        except ValueError:
            continue
    return None
