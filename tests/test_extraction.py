"""Entity extraction over text that looks like the reporting it will meet.

The passages below are written in the register Nigerian security reporting
actually uses, including the compound LGA names and the naira shorthand that
break naive matching.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pau_risk.nlp.extract import (
    Gazetteer,
    extract,
    find_lgas,
    normalise,
    parse_money,
    score_threat,
)


@pytest.fixture
def gazetteer():
    registry = pd.DataFrame(
        [
            {"lga_code": "NG021024", "lga_name": "Wasagu/Danko", "state_name": "Kebbi"},
            {"lga_code": "NG019001", "lga_name": "Birnin-Gwari", "state_name": "Kaduna"},
            {"lga_code": "NG036012", "lga_name": "Ifelodun", "state_name": "Osun"},
            {"lga_code": "NG024008", "lga_name": "Ifelodun", "state_name": "Kwara"},
            {"lga_code": "NG008021", "lga_name": "Maiduguri", "state_name": "Borno"},
        ]
    )
    return Gazetteer.from_registry(registry)


def test_compound_lga_names_match_in_either_form(gazetteer):
    for surface in ("Wasagu/Danko", "Wasagu Danko", "Danko"):
        matches = dict(find_lgas(f"Attack reported in {surface} LGA of Kebbi State", gazetteer))
        assert "NG021024" in matches


def test_hyphenated_names_match_without_the_hyphen(gazetteer):
    matches = dict(find_lgas("Bandits blocked the Birnin Gwari road", gazetteer))
    assert "NG019001" in matches


def test_ambiguous_name_is_resolved_by_the_state(gazetteer):
    resolved = dict(find_lgas("Violence in Ifelodun, Kwara State", gazetteer))
    assert resolved["NG024008"] > resolved.get("NG036012", 0)


def test_ambiguous_name_without_a_state_returns_both_at_low_confidence(gazetteer):
    matches = dict(find_lgas("Violence in Ifelodun", gazetteer))
    assert set(matches) == {"NG024008", "NG036012"}
    assert all(confidence < 0.6 for confidence in matches.values())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a ransom of N15 million was paid", 15_000_000),
        ("they demanded ₦2.5m", 2_500_000),
        ("the family paid 500,000 naira", 500_000),
        ("NGN 1.2 billion in levies", 1_200_000_000),
        ("the convoy of 20 vehicles", None),
        ("on 12 January 2024", None),
    ],
)
def test_money_extraction(text, expected):
    assert parse_money(text) == expected


def test_threat_score_rises_with_threat_vocabulary():
    quiet, _ = score_threat("The market reopened and traders returned to the town.")
    active, drivers = score_threat(
        "Gunmen on motorcycles ambushed travellers and abducted fourteen people for ransom."
    )
    assert active > quiet
    assert "abducted" in drivers or "ransom" in drivers


def test_negation_suppresses_a_threat_term():
    plain, _ = score_threat("Residents said bandits attacked the village.")
    denied, _ = score_threat("The police denied that bandits attacked the village.")
    assert denied < plain


def test_deescalation_lowers_the_score():
    ongoing, _ = score_threat("The abducted students remain in captivity.")
    resolved, _ = score_threat("The abducted students were released and reunited with families.")
    assert resolved < ongoing


def test_full_extraction_pulls_every_field(gazetteer):
    text = (
        "Armed bandits mounted a roadblock along the Birnin Gwari highway in Kaduna State "
        "on Tuesday and abducted at least 14 travellers. Residents said the attackers "
        "demanded a ransom of N15 million. Three people were killed in the attack."
    )
    facts = extract(text, gazetteer)
    assert "NG019001" in facts.lga_codes
    assert facts.ransom_ngn == 15_000_000
    assert facts.victims == 14
    assert facts.fatalities == 3
    assert facts.threat_score > 0.3


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("three people were killed", 3),
        ("Fourteen residents were killed in the raid", 14),
        ("17 villagers died", 17),
        ("two herders shot dead", 2),
        ("the market reopened", None),
    ],
)
def test_casualty_counts_in_digits_and_words(text, expected):
    from pau_risk.nlp.extract import FATALITY_PATTERN, parse_count

    assert parse_count(text, FATALITY_PATTERN) == expected


def test_normalisation_strips_accents_and_punctuation():
    assert normalise("Wasagu/Danko, Kebbi!") == "wasagu/danko kebbi"
