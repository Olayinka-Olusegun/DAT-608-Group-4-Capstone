"""Assemble the LGA by week feature panel.

The unit of analysis is one LGA in one ISO week, and the question asked of every
row is whether a banditry or kidnapping event occurs in the seven days that begin
on that Monday. Every feature on the row is therefore computed from data with a
timestamp strictly before that Monday. That rule is enforced structurally, by
taking window sums out of an exclusive prefix cumulative sum, rather than by
remembering to shift a rolling window, because an off by one here would leak the
answer into the question and produce metrics that cannot be reproduced in service.

Six families of feature are built.

Recent history, as lagged counts of events and fatalities over widening windows,
tells the model how active an LGA has been.

Neighbour history applies the same windows to the adjacency weighted sum over
neighbouring LGAs, which is what lets the model see displacement of effort into a
previously quiet area.

State operations are counted separately from violence, because a military
operation is a leading indicator: pressure applied in one LGA tends to relocate
armed groups into the next.

Hawkes intensity contributes the fitted decomposition of current risk into
background, self excitation and neighbour excitation.

Calendar terms carry the festival, season and school session effects.

Attention and chatter measure how much reporting and social traffic names the
LGA, relative to its own recent baseline rather than a national threshold.

The last two families depend on sources whose history is short, so every feature
is checked for coverage across the training window before it is allowed into the
model matrix. A feature that is almost always zero in training but populated at
serving time would be actively harmful, and the gate is what stops that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..logging_utils import get_logger
from ..reference.calendar_ng import calendar_features
from .hawkes import HawkesModel, neighbours_from_adjacency

LOGGER = get_logger(__name__)

TARGET_CLASS = "banditry_kidnapping"
OPERATION_CLASS = "state_operation"
COVERAGE_FLOOR = 0.01  # a feature must be non-zero on at least this share of training rows


FEATURE_LABELS: dict[str, str] = {
    "hawkes_background": "Baseline rate for this LGA",
    "hawkes_self": "Self-excitation from recent local attacks",
    "hawkes_neighbour": "Contagion from attacks in adjacent LGAs",
    "hawkes_total": "Total Hawkes intensity",
    "hawkes_self_share": "Share of intensity that is locally generated",
    "hawkes_neighbour_share": "Share of intensity arriving from neighbours",
    "hawkes_expected_events": "Hawkes expected events over the next 7 days",
    "hawkes_prob_any": "Hawkes probability of any event in 7 days",
    "own_events_1w": "Attacks here last week",
    "own_events_2w": "Attacks here in the last 2 weeks",
    "own_events_4w": "Attacks here in the last month",
    "own_events_8w": "Attacks here in the last 2 months",
    "own_events_12w": "Attacks here in the last quarter",
    "own_events_26w": "Attacks here in the last half year",
    "own_events_52w": "Attacks here in the last year",
    "own_fatalities_4w": "Deaths here in the last month",
    "own_fatalities_12w": "Deaths here in the last quarter",
    "own_fatalities_52w": "Deaths here in the last year",
    "nb_events_1w": "Attacks in adjacent LGAs last week",
    "nb_events_4w": "Attacks in adjacent LGAs in the last month",
    "nb_events_12w": "Attacks in adjacent LGAs in the last quarter",
    "nb_fatalities_4w": "Deaths in adjacent LGAs in the last month",
    "own_ops_4w": "Military operations here in the last month",
    "own_ops_12w": "Military operations here in the last quarter",
    "nb_ops_4w": "Active military operation in an adjacent LGA",
    "nb_ops_12w": "Military operations nearby in the last quarter",
    "weeks_since_event": "Weeks since the last attack here",
    "weeks_since_nb_event": "Weeks since the last attack next door",
    "state_events_4w": "Attacks across the state in the last month",
    "state_ops_4w": "Military operations across the state in the last month",
    "hist_rate": "Long-run attack rate for this LGA",
    "actors_26w": "Distinct armed actors active here or nearby",
    "cal_festival_within_horizon": "Festival or public holiday within 7 days",
    "cal_dry_season": "Dry season, when forest corridors are passable",
    "cal_school_in_session": "Schools in session",
    "cal_week_of_year": "Week of the year",
    "cal_month": "Month",
    "area_sqkm": "Size of the LGA",
    "n_neighbours": "Number of adjacent LGAs",
    "doc_mentions_4w": "Reports naming this LGA in the last month",
    "doc_ransom_12w": "Recent ransom payment reported nearby",
    "chatter_volume_1w": "Social chatter volume last week",
    "chatter_escalation": "Escalating chatter volume against own baseline",
    "chatter_threat_1w": "Peak threat score in chatter last week",
}


@dataclass
class PanelResult:
    frame: pd.DataFrame
    model_features: list[str]
    display_only: list[str]
    hawkes_parameters: dict[str, float]
    coverage: pd.DataFrame


def _week_grid(start: date, end: date) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="W-MON")


def _matrix(
    events: pd.DataFrame, codes: Sequence[str], weeks: pd.DatetimeIndex, value: str | None
) -> np.ndarray:
    """Counts, or a summed column, laid out as LGA by week."""
    index = {code: position for position, code in enumerate(codes)}
    week_index = {stamp: position for position, stamp in enumerate(weeks)}
    matrix = np.zeros((len(codes), len(weeks)), dtype=float)
    if events.empty:
        return matrix
    bucket = events["event_date"].dt.to_period("W-SUN").dt.start_time
    amounts = events[value].to_numpy(dtype=float) if value else np.ones(len(events))
    for code, stamp, amount in zip(events["lga_code"].to_numpy(), bucket, amounts):
        row, column = index.get(code), week_index.get(stamp)
        if row is not None and column is not None:
            matrix[row, column] += amount
    return matrix


def _prefix(matrix: np.ndarray) -> np.ndarray:
    """Exclusive cumulative sum along the time axis."""
    return np.concatenate(
        [np.zeros((matrix.shape[0], 1)), np.cumsum(matrix, axis=1)], axis=1
    )


def _window(prefix: np.ndarray, weeks: int) -> np.ndarray:
    """Sum over the ``weeks`` weeks that end immediately before each week."""
    n_weeks = prefix.shape[1] - 1
    out = np.zeros((prefix.shape[0], n_weeks), dtype=float)
    for column in range(n_weeks):
        lower = max(0, column - weeks)
        out[:, column] = prefix[:, column] - prefix[:, lower]
    return out


def _weeks_since(matrix: np.ndarray, censor: int) -> np.ndarray:
    """Weeks elapsed since the most recent non-zero week strictly before each week."""
    n_rows, n_weeks = matrix.shape
    out = np.full((n_rows, n_weeks), float(censor))
    last_seen = np.full(n_rows, -1, dtype=int)
    for column in range(n_weeks):
        gap = np.where(last_seen >= 0, column - last_seen, censor)
        out[:, column] = np.minimum(gap, censor)
        active = matrix[:, column] > 0
        last_seen[active] = column
    return out


def _adjacency_operator(
    adjacency: pd.DataFrame, codes: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalised neighbour weights as a dense operator, plus the degree vector."""
    index = {code: position for position, code in enumerate(codes)}
    operator = np.zeros((len(codes), len(codes)), dtype=float)
    for row in adjacency.itertuples():
        source, target = index.get(row.lga_code), index.get(row.neighbour_code)
        if source is not None and target is not None:
            operator[source, target] = float(row.weight)
    degree = (operator > 0).sum(axis=1).astype(float)
    return operator, degree


def build_panel(
    incidents: pd.DataFrame,
    registry: pd.DataFrame,
    adjacency: pd.DataFrame,
    documents: pd.DataFrame | None = None,
    chatter: pd.DataFrame | None = None,
    settings: Settings | None = None,
) -> PanelResult:
    settings = settings or get_settings()
    feature_cfg = settings.section("features")
    hawkes_cfg = settings.section("hawkes")
    model_cfg = settings.section("model")

    codes = registry["lga_code"].tolist()
    weeks = _week_grid(
        date.fromisoformat(str(feature_cfg["panel_start"])),
        date.fromisoformat(str(feature_cfg["panel_end"])),
    )
    LOGGER.info("panel grid: %d LGAs by %d weeks", len(codes), len(weeks))

    incidents = incidents[incidents["lga_code"].notna()].copy()
    incidents["event_date"] = pd.to_datetime(incidents["event_date"])
    target_events = incidents[incidents["event_class"] == TARGET_CLASS]
    operations = incidents[incidents["event_class"] == OPERATION_CLASS]

    counts = _matrix(target_events, codes, weeks, None)
    fatalities = _matrix(target_events, codes, weeks, "fatalities")
    ops = _matrix(operations, codes, weeks, None)

    operator, degree = _adjacency_operator(adjacency, codes)
    neighbour_counts = operator @ counts
    neighbour_fatalities = operator @ fatalities
    neighbour_ops = operator @ ops

    prefix_counts = _prefix(counts)
    prefix_fatalities = _prefix(fatalities)
    prefix_ops = _prefix(ops)
    prefix_nb_counts = _prefix(neighbour_counts)
    prefix_nb_fatalities = _prefix(neighbour_fatalities)
    prefix_nb_ops = _prefix(neighbour_ops)

    columns: dict[str, np.ndarray] = {}
    for window in feature_cfg["lag_weeks"]:
        columns[f"own_events_{window}w"] = _window(prefix_counts, window)
    for window in (4, 12, 52):
        columns[f"own_fatalities_{window}w"] = _window(prefix_fatalities, window)
    for window in (4, 12):
        columns[f"own_ops_{window}w"] = _window(prefix_ops, window)
    for window in feature_cfg["neighbour_lag_weeks"]:
        columns[f"nb_events_{window}w"] = _window(prefix_nb_counts, window)
    columns["nb_fatalities_4w"] = _window(prefix_nb_fatalities, 4)
    for window in (4, 12):
        columns[f"nb_ops_{window}w"] = _window(prefix_nb_ops, window)

    censor = int(feature_cfg["censor_weeks"])
    columns["weeks_since_event"] = _weeks_since(counts, censor)
    columns["weeks_since_nb_event"] = _weeks_since(neighbour_counts, censor)

    # Expanding historical rate, computed causally: events per week up to the
    # week before, which is the natural prior for an LGA with a thin history.
    elapsed = np.arange(1, len(weeks) + 1, dtype=float)
    columns["hist_rate"] = prefix_counts[:, :-1] / elapsed

    columns.update(_state_aggregates(registry, codes, counts, ops, weeks))
    columns["actors_26w"] = _actor_diversity(target_events, codes, weeks, operator)

    static = registry.set_index("lga_code")
    columns["area_sqkm"] = np.repeat(
        static.loc[codes, "area_sqkm"].to_numpy(dtype=float)[:, None], len(weeks), axis=1
    )
    columns["n_neighbours"] = np.repeat(degree[:, None], len(weeks), axis=1)

    frame = _to_long(columns, codes, weeks)
    frame = frame.merge(
        registry[["lga_code", "lga_name", "state_name", "zone"]], on="lga_code", how="left"
    )

    frame = _attach_calendar(frame, settings.horizon_days)
    frame, hawkes_parameters = _attach_hawkes(
        frame, target_events, adjacency, codes, weeks, hawkes_cfg, settings
    )
    frame = _attach_attention(frame, documents, chatter, codes, weeks)

    label_matrix = (counts > 0).astype(int)
    labels = pd.DataFrame(label_matrix, index=codes, columns=weeks).stack().rename("label")
    labels.index.names = ["lga_code", "week_start"]
    frame = frame.merge(labels.reset_index(), on=["lga_code", "week_start"], how="left")
    frame["label"] = frame["label"].fillna(0).astype(int)

    model_features, display_only, coverage = _gate_features(
        frame, train_end=date.fromisoformat(str(model_cfg["train_end"]))
    )

    LOGGER.info(
        "panel built: %d rows, %d model features, %d display-only, positive rate %.3f%%",
        len(frame),
        len(model_features),
        len(display_only),
        100.0 * frame["label"].mean(),
    )
    return PanelResult(
        frame=frame,
        model_features=model_features,
        display_only=display_only,
        hawkes_parameters=hawkes_parameters,
        coverage=coverage,
    )


# ------------------------------------------------------------------ helpers
def _to_long(
    columns: dict[str, np.ndarray], codes: Sequence[str], weeks: pd.DatetimeIndex
) -> pd.DataFrame:
    n_codes, n_weeks = len(codes), len(weeks)
    frame = pd.DataFrame(
        {
            "lga_code": np.repeat(np.asarray(codes), n_weeks),
            "week_start": np.tile(weeks.to_numpy(), n_codes),
        }
    )
    for name, matrix in columns.items():
        frame[name] = matrix.reshape(-1)
    return frame


def _state_aggregates(
    registry: pd.DataFrame,
    codes: Sequence[str],
    counts: np.ndarray,
    ops: np.ndarray,
    weeks: pd.DatetimeIndex,
) -> dict[str, np.ndarray]:
    states = registry.set_index("lga_code").loc[codes, "state_name"].to_numpy()
    output: dict[str, np.ndarray] = {}
    for name, matrix in (("state_events_4w", counts), ("state_ops_4w", ops)):
        totals = np.zeros_like(matrix)
        for state in np.unique(states):
            mask = states == state
            totals[mask] = matrix[mask].sum(axis=0)
        output[name] = _window(_prefix(totals), 4)
    return output


def _actor_diversity(
    events: pd.DataFrame, codes: Sequence[str], weeks: pd.DatetimeIndex, operator: np.ndarray
) -> np.ndarray:
    """Count distinct named armed actors seen in the LGA or its neighbours.

    A rise in the number of separate groups operating in an area is a different
    signal from a rise in the number of attacks: it points to competition or to
    an area becoming a corridor, both of which precede escalation.
    """
    presence = np.zeros((len(codes), len(weeks)), dtype=float)
    if events.empty:
        return presence
    index = {code: position for position, code in enumerate(codes)}
    week_index = {stamp: position for position, stamp in enumerate(weeks)}
    bucket = events["event_date"].dt.to_period("W-SUN").dt.start_time

    seen: dict[tuple[int, int], set[str]] = {}
    actors = events["actor_secondary"].fillna(events["actor_primary"]).fillna("unknown")
    for code, stamp, actor in zip(events["lga_code"].to_numpy(), bucket, actors):
        row, column = index.get(code), week_index.get(stamp)
        if row is None or column is None:
            continue
        seen.setdefault((row, column), set()).add(str(actor))
    for (row, column), names in seen.items():
        presence[row, column] = len(names)

    combined = presence + operator @ presence
    return _window(_prefix(combined), 26)


def _attach_calendar(frame: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    unique_weeks = pd.DatetimeIndex(frame["week_start"].unique())
    rows = []
    for stamp in unique_weeks:
        values = calendar_features(stamp.date(), horizon_days)
        values.pop("cal_festival_name", None)
        values["week_start"] = stamp
        rows.append(values)
    return frame.merge(pd.DataFrame(rows), on="week_start", how="left")


def _attach_hawkes(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    adjacency: pd.DataFrame,
    codes: Sequence[str],
    weeks: pd.DatetimeIndex,
    hawkes_cfg: dict,
    settings: Settings,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model = HawkesModel(neighbours=neighbours_from_adjacency(adjacency), codes=codes)
    fit_end = pd.Timestamp(str(hawkes_cfg["fit_end"]))
    origin = weeks[0]
    training_events = events[events["event_date"] <= fit_end]
    model.load_events(training_events, origin=origin, horizon_end=fit_end)
    parameters = model.fit(
        decay_init=float(hawkes_cfg["decay_init"]),
        max_iterations=int(hawkes_cfg["max_iterations"]),
    )
    # Parameters are estimated on the training window only, then the full event
    # history is loaded so that intensity at later dates conditions on everything
    # observable at that date without the parameters having seen the test period.
    model.load_events(events, origin=origin, horizon_end=weeks[-1])
    model.parameters = parameters
    intensity = model.intensity_frame(weeks, horizon_days=settings.horizon_days)
    merged = frame.merge(intensity, on=["lga_code", "week_start"], how="left")
    return merged, parameters.as_dict()


def _attach_attention(
    frame: pd.DataFrame,
    documents: pd.DataFrame | None,
    chatter: pd.DataFrame | None,
    codes: Sequence[str],
    weeks: pd.DatetimeIndex,
) -> pd.DataFrame:
    frame["doc_mentions_4w"] = 0.0
    frame["doc_ransom_12w"] = 0.0
    frame["chatter_volume_1w"] = 0.0
    frame["chatter_threat_1w"] = 0.0
    frame["chatter_escalation"] = 0.0

    if documents is not None and not documents.empty:
        mentions, ransom = _document_matrices(documents, codes, weeks)
        frame = _merge_matrix(frame, _window(_prefix(mentions), 4), codes, weeks, "doc_mentions_4w")
        frame = _merge_matrix(frame, _window(_prefix(ransom), 12), codes, weeks, "doc_ransom_12w")

    if chatter is not None and not chatter.empty:
        volume, threat = _chatter_matrices(chatter, codes, weeks)
        frame = _merge_matrix(frame, _window(_prefix(volume), 1), codes, weeks, "chatter_volume_1w")
        frame = _merge_matrix(frame, _window(_prefix(threat), 1), codes, weeks, "chatter_threat_1w")
        recent = _window(_prefix(volume), 1)
        baseline = _window(_prefix(volume), 8) / 8.0
        escalation = np.divide(
            recent, baseline, out=np.zeros_like(recent), where=baseline > 0
        )
        frame = _merge_matrix(frame, escalation, codes, weeks, "chatter_escalation")
    return frame


def _document_matrices(
    documents: pd.DataFrame, codes: Sequence[str], weeks: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    index = {code: position for position, code in enumerate(codes)}
    week_index = {stamp: position for position, stamp in enumerate(weeks)}
    mentions = np.zeros((len(codes), len(weeks)), dtype=float)
    ransom = np.zeros((len(codes), len(weeks)), dtype=float)
    published = pd.to_datetime(documents["published_at"], errors="coerce")
    bucket = published.dt.to_period("W-SUN").dt.start_time
    for raw_codes, stamp, amount in zip(documents["lga_codes"], bucket, documents["ransom_ngn"]):
        column = week_index.get(stamp)
        if column is None:
            continue
        try:
            resolved = json.loads(raw_codes) if isinstance(raw_codes, str) else list(raw_codes or [])
        except (TypeError, ValueError):
            resolved = []
        for code in resolved:
            row = index.get(code)
            if row is None:
                continue
            mentions[row, column] += 1
            if amount and float(amount) > 0:
                ransom[row, column] += 1
    return mentions, ransom


def _chatter_matrices(
    chatter: pd.DataFrame, codes: Sequence[str], weeks: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    index = {code: position for position, code in enumerate(codes)}
    week_index = {stamp: position for position, stamp in enumerate(weeks)}
    volume = np.zeros((len(codes), len(weeks)), dtype=float)
    threat = np.zeros((len(codes), len(weeks)), dtype=float)
    posted = pd.to_datetime(chatter["posted_at"], errors="coerce", utc=True).dt.tz_localize(None)
    bucket = posted.dt.to_period("W-SUN").dt.start_time
    for code, stamp, score in zip(chatter["lga_code"], bucket, chatter["threat_score"]):
        row, column = index.get(code), week_index.get(stamp)
        if row is None or column is None:
            continue
        volume[row, column] += 1
        threat[row, column] = max(threat[row, column], float(score or 0.0))
    return volume, threat


def _merge_matrix(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    codes: Sequence[str],
    weeks: pd.DatetimeIndex,
    column: str,
) -> pd.DataFrame:
    values = pd.DataFrame(matrix, index=codes, columns=weeks).stack().rename(column)
    values.index.names = ["lga_code", "week_start"]
    frame = frame.drop(columns=[column], errors="ignore")
    return frame.merge(values.reset_index(), on=["lga_code", "week_start"], how="left")


def _gate_features(
    frame: pd.DataFrame, train_end: date, recent_weeks: int = 26
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Decide which features are safe to train on.

    Two things disqualify a feature, and neither of them is rarity. A feature that
    fires on half a percent of rows is not weak, it is describing a rare event,
    and that is exactly the signal the model exists to find.

    The first disqualification is having no variation at all across the panel,
    which carries no information and only widens the search space.

    The second is training and serving skew. If a feature is close to empty
    across the training window but populated in recent weeks, its source started
    reporting part way through, so the model would learn a relationship from
    almost no examples and then meet a fully populated column in service. The
    social and document features behave this way whenever their connectors are
    newly credentialed, so the comparison is made explicitly rather than assumed.
    """
    excluded = {"lga_code", "week_start", "lga_name", "state_name", "zone", "label"}
    candidates = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    training = frame[frame["week_start"] <= pd.Timestamp(train_end)]
    cutoff = frame["week_start"].max() - pd.Timedelta(weeks=recent_weeks)
    recent = frame[frame["week_start"] > cutoff]

    rows = []
    for column in candidates:
        train_coverage = float((training[column].fillna(0) != 0).mean())
        recent_coverage = float((recent[column].fillna(0) != 0).mean())
        constant = float(frame[column].std(skipna=True) or 0.0) == 0.0
        skewed = train_coverage < COVERAGE_FLOOR and recent_coverage > 5 * train_coverage
        reason = "constant" if constant else ("train-serve skew" if skewed else "")
        rows.append(
            {
                "feature": column,
                "label": FEATURE_LABELS.get(column, column),
                "train_coverage": train_coverage,
                "recent_coverage": recent_coverage,
                "in_model": not (constant or skewed),
                "excluded_because": reason,
            }
        )
    coverage_frame = pd.DataFrame(rows).sort_values("train_coverage", ascending=False)
    model_features = coverage_frame.loc[coverage_frame["in_model"], "feature"].tolist()
    display_only = coverage_frame.loc[~coverage_frame["in_model"], "feature"].tolist()
    if display_only:
        LOGGER.info(
            "held out of the model: %s",
            ", ".join(
                f"{row.feature} ({row.excluded_because})"
                for row in coverage_frame.loc[~coverage_frame["in_model"]].itertuples()
            ),
        )
    return model_features, display_only, coverage_frame
