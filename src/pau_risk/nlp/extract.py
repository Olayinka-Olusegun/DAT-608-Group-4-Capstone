"""Entity extraction over the unstructured feeds.

Three things have to come out of a report, an advisory or a social post before it
can join the feature panel: which LGA it refers to, whether it describes the kind
of activity that precedes an attack, and how much money changed hands.

Place matching runs against a gazetteer built from the official LGA registry
rather than a fixed list, so it stays correct if boundaries are revised. Roughly
one LGA name in ten is shared between states, Ifelodun and Obi being the clearest
cases, so an ambiguous surface form is only resolved when the surrounding text
names a state; otherwise it is returned as a candidate set and the caller decides.

Scoring is lexicon based on purpose. A transformer classifier would score higher
on a benchmark, but the operational requirement here is that an analyst can read
why a post raised a score, and a weighted lexicon with explicit negation handling
is auditable in a way a fine-tuned encoder is not. The lexicon is the point of
extension, not the algorithm.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

# Weighted indicators of imminent or realised armed activity. Positive weights
# raise the threat score; the magnitude reflects how far ahead of an incident the
# term usually appears in reporting.
THREAT_LEXICON: dict[str, float] = {
    "gunmen": 0.8, "bandits": 0.9, "banditry": 0.9, "kidnap": 1.0, "kidnapped": 1.0,
    "abduct": 1.0, "abducted": 1.0, "abduction": 1.0, "ransom": 0.9, "hostage": 0.9,
    "attack": 0.7, "attacked": 0.7, "raid": 0.7, "ambush": 0.8, "invaded": 0.7,
    "herdsmen": 0.5, "militia": 0.6, "insurgent": 0.7, "terrorist": 0.7,
    "gunfire": 0.6, "shooting": 0.6, "killed": 0.6, "massacre": 0.9,
    "displaced": 0.4, "fleeing": 0.5, "roadblock": 0.6, "highway": 0.3,
    "motorcycles": 0.4, "reinforcement": 0.3, "levy": 0.5, "tax": 0.2,
    "warning": 0.4, "threat": 0.6, "alert": 0.5, "sighted": 0.6, "movement": 0.4,
    "regrouping": 0.8, "camp": 0.5, "forest": 0.4, "informant": 0.5,
    "school": 0.4, "students": 0.5, "pupils": 0.5, "worshippers": 0.5,
    "military operation": 0.5, "airstrike": 0.6, "troops": 0.4, "deployment": 0.4,
}

# Terms that indicate the situation is resolving rather than escalating.
DEESCALATION_LEXICON: dict[str, float] = {
    "released": -0.7, "rescued": -0.8, "freed": -0.7, "reunited": -0.6,
    "arrested": -0.4, "neutralised": -0.5, "neutralized": -0.5, "surrendered": -0.6,
    "peace": -0.4, "calm": -0.5, "restored": -0.5, "normalcy": -0.6,
}

NEGATIONS = {"no", "not", "never", "denied", "denies", "false", "rumour", "rumor"}

RANSOM_PATTERNS = [
    re.compile(
        r"(?:₦|N|NGN)\s?([\d,]+(?:\.\d+)?)\s*(million|m|billion|bn|b|thousand|k)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?\s*naira\b", re.IGNORECASE
    ),
]

MULTIPLIERS = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
}

# Casualty counts below about twenty are usually written as words in Nigerian
# reporting, and those are exactly the counts that matter for a kidnapping
# rather than a mass casualty attack, so both forms are matched.
WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "dozen": 12, "scores": 20, "several": 3,
}
_NUMBER = r"(\d{1,4}|" + "|".join(sorted(WORD_NUMBERS, key=len, reverse=True)) + r")"

# The nouns that appear between a count and the verb in Nigerian reporting.
# Occupational descriptions are as common as the generic ones, and headlines
# such as "Two herders shot dead in Plateau" are missed without them.
_PEOPLE = (
    r"persons?|people|residents?|villagers?|students?|pupils?|passengers?|"
    r"worshippers?|farmers?|herders?|herdsmen|travellers?|travelers?|traders?|"
    r"miners?|vigilantes?|soldiers?|troops|officers?|police\w*|civilians?|"
    r"women|children|schoolgirls?|schoolboys?|others?|victims?"
)

VICTIM_PATTERN = re.compile(rf"\b{_NUMBER}\s+(?:{_PEOPLE})\b", re.IGNORECASE)

FATALITY_PATTERN = re.compile(
    rf"\b{_NUMBER}\s+(?:{_PEOPLE})?\s*"
    r"(?:were\s+|was\s+|have\s+been\s+|has\s+been\s+)?"
    r"(?:killed|dead|died|slain|murdered|shot\s+dead|gunned\s+down)\b",
    re.IGNORECASE,
)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s/'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Gazetteer:
    """Surface forms mapped to LGA codes, with the state index used to disambiguate."""

    by_name: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    state_by_code: dict[str, str] = field(default_factory=dict)
    name_by_code: dict[str, str] = field(default_factory=dict)
    states: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_registry(cls, registry: pd.DataFrame) -> "Gazetteer":
        gazetteer = cls()
        for row in registry.itertuples():
            code = row.lga_code
            gazetteer.state_by_code[code] = row.state_name
            gazetteer.name_by_code[code] = row.lga_name
            for surface in cls._surface_forms(row.lga_name):
                gazetteer.by_name[surface].append(code)
            gazetteer.states[normalise(row.state_name)] = row.state_name
        gazetteer.states.setdefault("fct", "Federal Capital Territory")
        gazetteer.states.setdefault("abuja", "Federal Capital Territory")
        return gazetteer

    @staticmethod
    def _surface_forms(name: str) -> set[str]:
        base = normalise(name)
        forms = {base}
        # Compound names such as Wasagu/Danko are written several ways in reporting.
        if "/" in base:
            parts = [part.strip() for part in base.split("/") if len(part.strip()) > 3]
            forms.update(parts)
            forms.add(base.replace("/", " "))
        if "-" in base:
            forms.add(base.replace("-", " "))
        forms.add(re.sub(r"\b(east|west|north|south|central)\b", "", base).strip())
        return {form for form in forms if len(form) > 3}

    def is_ambiguous(self, surface: str) -> bool:
        return len(self.by_name.get(surface, [])) > 1


@lru_cache(maxsize=1)
def _cached_gazetteer_key() -> None:
    return None


def find_states(text: str, gazetteer: Gazetteer) -> set[str]:
    normalised = normalise(text)
    return {
        proper
        for surface, proper in gazetteer.states.items()
        if re.search(rf"\b{re.escape(surface)}\b", normalised)
    }


def find_lgas(text: str, gazetteer: Gazetteer) -> list[tuple[str, float]]:
    """Return LGA codes mentioned in the text with a confidence between 0 and 1."""
    normalised = normalise(text)
    mentioned_states = find_states(text, gazetteer)
    matches: dict[str, float] = {}

    for surface, codes in gazetteer.by_name.items():
        if not re.search(rf"\b{re.escape(surface)}\b", normalised):
            continue
        if len(codes) == 1:
            matches[codes[0]] = max(matches.get(codes[0], 0.0), 0.95)
            continue
        resolved = [
            code for code in codes if gazetteer.state_by_code[code] in mentioned_states
        ]
        if len(resolved) == 1:
            matches[resolved[0]] = max(matches.get(resolved[0], 0.0), 0.9)
        else:
            share = 0.5 / len(codes)
            for code in codes:
                matches[code] = max(matches.get(code, 0.0), share)
    return sorted(matches.items(), key=lambda item: -item[1])


def parse_money(text: str) -> float | None:
    """Extract the largest naira figure in the text, in naira."""
    best: float | None = None
    for pattern in RANSOM_PATTERNS:
        for match in pattern.finditer(text or ""):
            raw = match.group(1).replace(",", "")
            try:
                amount = float(raw)
            except ValueError:
                continue
            suffix = (match.group(2) or "").lower()
            amount *= MULTIPLIERS.get(suffix, 1.0)
            # Figures below ten thousand naira are almost never ransom demands and
            # are usually dates, counts or article identifiers picked up in error.
            if amount < 10_000:
                continue
            best = amount if best is None else max(best, amount)
    return best


def parse_count(text: str, pattern: re.Pattern[str]) -> int | None:
    values: list[int] = []
    for match in pattern.finditer(text or ""):
        token = match.group(1)
        if token.isdigit():
            values.append(int(token))
        else:
            resolved = WORD_NUMBERS.get(token.lower())
            if resolved is not None:
                values.append(resolved)
    return max(values) if values else None


def score_threat(text: str) -> tuple[float, list[str]]:
    """Score a passage and return the terms that drove the score.

    Returning the matched terms keeps the chatter feature explainable end to end:
    an analyst can trace a raised chatter score in the dashboard back to the exact
    vocabulary that produced it.
    """
    tokens = normalise(text).split()
    if not tokens:
        return 0.0, []

    total, drivers = 0.0, []
    joined = " ".join(tokens)
    for phrase, weight in THREAT_LEXICON.items():
        if " " not in phrase:
            continue
        if phrase in joined:
            total += weight
            drivers.append(phrase)

    for index, token in enumerate(tokens):
        weight = THREAT_LEXICON.get(token)
        if weight is None:
            continue
        window = tokens[max(0, index - 3) : index]
        if NEGATIONS & set(window):
            continue
        total += weight
        drivers.append(token)

    for token in tokens:
        weight = DEESCALATION_LEXICON.get(token)
        if weight is not None:
            total += weight
            drivers.append(token)

    # Squash to the unit interval so the feature is comparable across post lengths.
    length_scale = max(1.0, len(tokens) / 40.0)
    score = total / (length_scale + abs(total))
    return max(0.0, min(1.0, (score + 1) / 2 if score < 0 else score)), drivers[:8]


def score_sentiment(text: str) -> float:
    """A crude polarity in the range minus one to one, driven by the same lexicons."""
    tokens = set(normalise(text).split())
    negative = sum(1 for token in tokens if token in THREAT_LEXICON)
    positive = sum(1 for token in tokens if token in DEESCALATION_LEXICON)
    if not (negative or positive):
        return 0.0
    return (positive - negative) / (positive + negative)


@dataclass
class ExtractedFacts:
    lga_codes: list[str]
    confidence: float
    states: list[str]
    ransom_ngn: float | None
    victims: int | None
    fatalities: int | None
    threat_score: float
    sentiment: float
    drivers: list[str]


def extract(text: str, gazetteer: Gazetteer, top_k: int = 5) -> ExtractedFacts:
    matches = find_lgas(text, gazetteer)[:top_k]
    threat, drivers = score_threat(text)
    return ExtractedFacts(
        lga_codes=[code for code, _ in matches],
        confidence=matches[0][1] if matches else 0.0,
        states=sorted(find_states(text, gazetteer)),
        ransom_ngn=parse_money(text),
        victims=parse_count(text, VICTIM_PATTERN),
        fatalities=parse_count(text, FATALITY_PATTERN),
        threat_score=threat,
        sentiment=score_sentiment(text),
        drivers=drivers,
    )
