"""Real-time chatter from curated watchlists.

This is the only producer in the topology that can surface an indicator before an
incident happens, which is also why it is the one that has to be handled most
carefully. Posts are not evidence. A single account claiming movement on a road
is worth very little; a rise in the rate of independent accounts describing the
same corridor over three days is worth a great deal. The connector therefore
stores each post with its threat score and its matched vocabulary, and leaves the
aggregation to the feature layer, where volume and escalation are measured
against an LGA's own recent baseline rather than against a national threshold.

Two platforms are wired. X is read through the recent search endpoint, which
needs a bearer token and only reaches back seven days on the basic tier.
Telegram is read through the public channel export interface, which needs an
application identifier and hash. Neither runs without credentials, and the
connector reports that plainly instead of returning an empty result that would
look like quiet.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any, Iterable

import requests

from ..logging_utils import get_logger
from ..nlp.extract import Gazetteer, extract
from ..reference import boundaries
from .base import Availability, Connector, Readiness, stable_id

LOGGER = get_logger(__name__)

X_SEARCH_URL = "https://api.x.com/2/tweets/search/recent"

# Curated watchlist. Accounts and channels are the operational input a state
# security council would maintain; the query terms are the standing filter.
WATCHLIST_TERMS = (
    "bandits", "kidnap", "abduction", "gunmen", "ransom", "highway attack",
    "forest camp", "vigilante", "military operation",
)
WATCHLIST_ACCOUNTS = (
    "PoliceNG", "HQNigerianArmy", "NEMAnigeria", "HumAngle_", "Zagazola",
)
TELEGRAM_CHANNELS = ("nigeriasecuritytracker", "northwestsecurity")


class SocialConnector(Connector):
    name = "social"
    kind = "chatter"
    cadence = "streaming"
    description = "X and Telegram watchlists, scored for threat vocabulary"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._gazetteer: Gazetteer | None = None

    @property
    def gazetteer(self) -> Gazetteer:
        if self._gazetteer is None:
            self._gazetteer = Gazetteer.from_registry(boundaries().frame)
        return self._gazetteer

    def availability(self) -> Availability:
        if self.settings.env("X_BEARER_TOKEN"):
            return Availability(Readiness.READY, "X bearer token present")
        if self.settings.env("TELEGRAM_API_ID") and self.settings.env("TELEGRAM_API_HASH"):
            return Availability(Readiness.READY, "Telegram application credentials present")
        return Availability(
            Readiness.NEEDS_CREDENTIALS,
            "set X_BEARER_TOKEN, or TELEGRAM_API_ID with TELEGRAM_API_HASH",
        )

    # ---------------------------------------------------------------- fetch
    def fetch(self, since: date, until: date) -> list[dict[str, Any]]:
        posts: list[dict[str, Any]] = []
        if self.settings.env("X_BEARER_TOKEN"):
            posts.extend(self._fetch_x(since, until))
        if self.settings.env("TELEGRAM_API_ID"):
            posts.extend(self._fetch_telegram(since, until))
        return [self._score(post) for post in posts]

    def _fetch_x(self, since: date, until: date) -> Iterable[dict[str, Any]]:
        token = self.settings.env("X_BEARER_TOKEN")
        term_clause = " OR ".join(f'"{term}"' for term in WATCHLIST_TERMS)
        account_clause = " OR ".join(f"from:{handle}" for handle in WATCHLIST_ACCOUNTS)
        query = f"(({term_clause}) place_country:NG OR ({account_clause})) -is:retweet lang:en"

        collected: list[dict[str, Any]] = []
        next_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "query": query[:1024],
                "max_results": 100,
                "tweet.fields": "created_at,geo,text,author_id",
                "start_time": datetime.combine(since, time.min, timezone.utc).isoformat(),
                "end_time": datetime.combine(until, time.min, timezone.utc).isoformat(),
            }
            if next_token:
                params["next_token"] = next_token
            response = requests.get(
                X_SEARCH_URL, params=params, headers={"Authorization": f"Bearer {token}"}, timeout=60
            )
            if response.status_code != 200:
                LOGGER.info("X search returned %d", response.status_code)
                break
            payload = response.json()
            for item in payload.get("data", []):
                collected.append(
                    {
                        "platform": "x",
                        "posted_at": item.get("created_at"),
                        "body": item.get("text", ""),
                        "url": f"https://x.com/i/web/status/{item.get('id')}",
                    }
                )
            next_token = payload.get("meta", {}).get("next_token")
            if not next_token or len(collected) >= 1000:
                break
        return collected

    def _fetch_telegram(self, since: date, until: date) -> Iterable[dict[str, Any]]:
        """Read public channel history through Telethon when it is installed."""
        try:
            from telethon.sync import TelegramClient  # type: ignore[import-not-found]
        except ImportError:
            LOGGER.info("telethon is not installed, Telegram watchlist skipped")
            return []

        api_id = int(self.settings.env("TELEGRAM_API_ID") or 0)
        api_hash = self.settings.env("TELEGRAM_API_HASH") or ""
        collected: list[dict[str, Any]] = []
        session_path = str(self.settings.paths.raw / "telegram.session")
        with TelegramClient(session_path, api_id, api_hash) as client:
            for channel in TELEGRAM_CHANNELS:
                try:
                    for message in client.iter_messages(channel, limit=500):
                        if not message.text:
                            continue
                        stamp = message.date.date()
                        if not (since <= stamp <= until):
                            continue
                        collected.append(
                            {
                                "platform": "telegram",
                                "posted_at": message.date.isoformat(),
                                "body": message.text,
                                "url": f"https://t.me/{channel}/{message.id}",
                            }
                        )
                except Exception as exc:  # noqa: BLE001 - a private channel must not stop the run
                    LOGGER.info("telegram channel %s unavailable (%s)", channel, exc.__class__.__name__)
        return collected

    def _score(self, post: dict[str, Any]) -> dict[str, Any]:
        facts = extract(post["body"], self.gazetteer)
        return self._chatter(
            chatter_id=stable_id(self.name, post.get("url"), post.get("posted_at")),
            platform=post["platform"],
            posted_at=post["posted_at"],
            lga_code=facts.lga_codes[0] if facts.lga_codes else None,
            body=post["body"][:2000],
            sentiment=facts.sentiment,
            threat_score=facts.threat_score,
            url=post.get("url"),
        )
