"""Nigerian calendar signals used as leading indicators.

Three calendar effects are encoded, each with an operational rationale rather
than a statistical one.

Public holidays and religious festivals concentrate travel and cash movement,
and the brief names a festival inside the seven day horizon as one of the
drivers a security council should see. Christian dates are computed with the
Gregorian computus. Islamic dates are converted with the tabular Islamic
calendar, which tracks observed sightings to within roughly a day; the flag is
therefore widened by one day on each side rather than being claimed as exact.

Season matters because dry season passability opens forest corridors between
Zamfara, Katsina, Kaduna and Niger, and closes them again once the rains make
the same tracks impassable.

School terms matter because mass school abductions can only happen while
boarding houses are occupied. The term windows follow the federal unified
academic calendar and are treated as an approximation, not a per state timetable.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

ISLAMIC_EPOCH_JDN = 1948439  # 1 Muharram 1 AH in the tabular calendar


def easter_sunday(year: int) -> date:
    """Anonymous Gregorian computus."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lunar = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lunar) // 451
    month, day = divmod(h + lunar - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _gregorian_to_jdn(value: date) -> int:
    a = (14 - value.month) // 12
    y = value.year + 4800 - a
    m = value.month + 12 * a - 3
    return (
        value.day
        + (153 * m + 2) // 5
        + 365 * y
        + y // 4
        - y // 100
        + y // 400
        - 32045
    )


def _jdn_to_gregorian(jdn: int) -> date:
    a = jdn + 32044
    b, c = divmod(4 * a + 3, 146097)
    d = c // 4
    e, f = divmod(4 * d + 3, 1461)
    g = f // 4
    h, i = divmod(5 * g + 2, 153)
    day = i // 5 + 1
    month = (h + 2) % 12 + 1
    year = 100 * b + e - 4800 + (h + 2) // 12
    return date(year, month, day)


def _islamic_to_gregorian(hijri_year: int, hijri_month: int, hijri_day: int) -> date:
    jdn = (
        (hijri_day - 1)
        + (29 * (hijri_month - 1) + (hijri_month // 2))
        + (hijri_year - 1) * 354
        + (3 + 11 * hijri_year) // 30
        + ISLAMIC_EPOCH_JDN
    )
    return _jdn_to_gregorian(jdn)


def _hijri_years_overlapping(year: int) -> list[int]:
    start = _gregorian_to_jdn(date(year, 1, 1))
    approximate = int((start - ISLAMIC_EPOCH_JDN) / 354.367) + 1
    return [approximate - 1, approximate, approximate + 1]


@lru_cache(maxsize=64)
def holidays_for_year(year: int) -> dict[date, str]:
    """Return the national holiday and festival dates observed in a given year."""
    easter = easter_sunday(year)
    observances: dict[date, str] = {
        date(year, 1, 1): "New Year",
        easter - timedelta(days=2): "Good Friday",
        easter + timedelta(days=1): "Easter Monday",
        date(year, 5, 1): "Workers Day",
        date(year, 10, 1): "Independence Day",
        date(year, 12, 25): "Christmas",
        date(year, 12, 26): "Boxing Day",
    }
    # Democracy Day moved from 29 May to 12 June with effect from 2019.
    observances[date(year, 6, 12) if year >= 2019 else date(year, 5, 29)] = "Democracy Day"

    for hijri_year in _hijri_years_overlapping(year):
        candidates = {
            _islamic_to_gregorian(hijri_year, 10, 1): "Eid al-Fitr",
            _islamic_to_gregorian(hijri_year, 12, 10): "Eid al-Adha",
            _islamic_to_gregorian(hijri_year, 3, 12): "Eid al-Mawlid",
        }
        for observed, label in candidates.items():
            if observed.year == year:
                observances[observed] = label
    return observances


def festival_within(reference: date, days: int = 7) -> tuple[int, str | None]:
    """Flag whether a festival falls inside the forecast horizon.

    The window is widened by one day on either side to absorb the uncertainty in
    the tabular Islamic conversion and the practice of declaring public holidays
    on the following working day.
    """
    window_start = reference - timedelta(days=1)
    window_end = reference + timedelta(days=days + 1)
    for year in {window_start.year, window_end.year}:
        for observed, label in holidays_for_year(year).items():
            if window_start <= observed <= window_end:
                return 1, label
    return 0, None


def is_dry_season(reference: date) -> int:
    """Dry season runs from November to March across the northern zones."""
    return int(reference.month in (11, 12, 1, 2, 3))


def school_in_session(reference: date) -> int:
    """Approximate the federal unified academic calendar."""
    month, day = reference.month, reference.day
    if month in (8,):
        return 0
    if month == 7 and day > 20:
        return 0
    if month == 12 and day > 15:
        return 0
    if month == 1 and day < 8:
        return 0
    if month == 4 and 1 <= day <= 14:
        return 0
    return 1


def calendar_features(reference: date, horizon_days: int = 7) -> dict[str, float | int | str | None]:
    festival_flag, festival_name = festival_within(reference, horizon_days)
    return {
        "cal_festival_within_horizon": festival_flag,
        "cal_festival_name": festival_name,
        "cal_dry_season": is_dry_season(reference),
        "cal_school_in_session": school_in_session(reference),
        "cal_week_of_year": reference.isocalendar().week,
        "cal_month": reference.month,
    }
