"""The threat actor graph: who is targeting whom.

The risk score says where. It does not say who, and a council that knows an LGA
is severe still has to decide whether it is facing a mobile bandit group that
raids and withdraws, an insurgent group holding ground, or a communal conflict
that a security deployment can inflame rather than contain. Those call for
different responses, so the actor structure is modelled separately from the score.

The graph has three kinds of edge, all built from coded event records rather than
inferred from text.

An actor to LGA edge means a named armed actor has been recorded operating in
that LGA, weighted by how many events and how recently. Recency is applied as an
exponential decay with a half life of a year, because a group that last appeared
in 2015 tells you much less about this week than one that appeared in March.

An actor to state edge is the same relationship rolled up, which is the level at
which a state security council actually allocates.

An actor to actor edge records the two sides of a non-state confrontation. These
are the rivalry pairs, and they matter because competition between armed groups
over the same territory is one of the reliable precursors of escalation against
the civilians living in it.

Centrality is computed on the actor projection. Degree identifies the groups
operating across the widest footprint, which is a different and often more useful
question than which group has killed the most people.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

HALF_LIFE_DAYS = 365.0
UNINFORMATIVE_ACTORS = {
    "civilians", "unknown", "", "none", "civilians (nigeria)",
}


@dataclass
class ThreatGraph:
    edges: pd.DataFrame
    actors: pd.DataFrame
    rivalries: pd.DataFrame

    def for_lga(self, lga_code: str, top_k: int = 5) -> pd.DataFrame:
        subset = self.edges[
            (self.edges["target_kind"] == "lga") & (self.edges["target"] == lga_code)
        ]
        return subset.nlargest(top_k, "weight")

    def for_state(self, state_name: str, top_k: int = 8) -> pd.DataFrame:
        subset = self.edges[
            (self.edges["target_kind"] == "state") & (self.edges["target"] == state_name)
        ]
        return subset.nlargest(top_k, "weight")


def _clean_actor(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() in UNINFORMATIVE_ACTORS:
        return None
    return text


def _recency_weight(event_date: pd.Timestamp, reference: pd.Timestamp) -> float:
    age_days = max((reference - event_date).days, 0)
    return math.pow(0.5, age_days / HALF_LIFE_DAYS)


def build(
    incidents: pd.DataFrame,
    registry: pd.DataFrame,
    run_id: str,
    reference_date: date | None = None,
    lookback_years: int = 5,
) -> ThreatGraph:
    frame = incidents[incidents["lga_code"].notna()].copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    reference = pd.Timestamp(reference_date or frame["event_date"].max())
    frame = frame[frame["event_date"] >= reference - pd.Timedelta(days=365 * lookback_years)]

    # The armed actor is normally the second party in a state based record and the
    # first in one sided violence, so both are considered and civilians dropped.
    frame["actor"] = frame["actor_secondary"].map(_clean_actor)
    frame.loc[frame["actor"].isna(), "actor"] = frame["actor_primary"].map(_clean_actor)
    frame = frame[frame["actor"].notna()]
    if frame.empty:
        LOGGER.info("no named actors in the window, threat graph is empty")
        empty = pd.DataFrame()
        return ThreatGraph(edges=empty, actors=empty, rivalries=empty)

    frame["recency"] = frame["event_date"].map(lambda stamp: _recency_weight(stamp, reference))
    states = registry.set_index("lga_code")["state_name"]
    frame["state_name"] = frame["lga_code"].map(states)

    edges = pd.concat(
        [
            _aggregate(frame, "lga_code", "lga", run_id),
            _aggregate(frame, "state_name", "state", run_id),
        ],
        ignore_index=True,
    )

    actors = (
        edges[edges["target_kind"] == "lga"]
        .groupby("actor")
        .agg(
            lgas=("target", "nunique"),
            events=("events", "sum"),
            fatalities=("fatalities", "sum"),
            weight=("weight", "sum"),
            last_seen=("last_seen", "max"),
        )
        .reset_index()
        .sort_values("weight", ascending=False)
    )
    actors["run_id"] = run_id
    actors = _add_centrality(actors, edges)

    rivalries = _rivalries(frame, run_id)
    LOGGER.info(
        "threat graph: %d edges, %d actors, %d rivalry pairs",
        len(edges),
        len(actors),
        len(rivalries),
    )
    return ThreatGraph(edges=edges, actors=actors, rivalries=rivalries)


def _aggregate(frame: pd.DataFrame, column: str, kind: str, run_id: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["actor", column])
        .agg(
            events=("event_id", "count"),
            fatalities=("fatalities", "sum"),
            first_seen=("event_date", "min"),
            last_seen=("event_date", "max"),
            weight=("recency", "sum"),
        )
        .reset_index()
        .rename(columns={column: "target"})
    )
    grouped["target_kind"] = kind
    grouped["run_id"] = run_id
    grouped["first_seen"] = grouped["first_seen"].dt.date.astype(str)
    grouped["last_seen"] = grouped["last_seen"].dt.date.astype(str)
    grouped["fatalities"] = grouped["fatalities"].fillna(0).astype(int)
    return grouped[
        [
            "run_id", "actor", "target_kind", "target", "events",
            "fatalities", "first_seen", "last_seen", "weight",
        ]
    ]


def _rivalries(frame: pd.DataFrame, run_id: str) -> pd.DataFrame:
    pairs = frame[
        frame["actor_primary"].map(_clean_actor).notna()
        & frame["actor_secondary"].map(_clean_actor).notna()
    ].copy()
    if pairs.empty:
        return pd.DataFrame()
    pairs["side_a"] = pairs["actor_primary"].map(_clean_actor)
    pairs["side_b"] = pairs["actor_secondary"].map(_clean_actor)
    ordered = pairs.apply(
        lambda row: tuple(sorted((row["side_a"], row["side_b"]))), axis=1
    )
    pairs["pair_a"] = [pair[0] for pair in ordered]
    pairs["pair_b"] = [pair[1] for pair in ordered]
    grouped = (
        pairs.groupby(["pair_a", "pair_b"])
        .agg(
            events=("event_id", "count"),
            fatalities=("fatalities", "sum"),
            weight=("recency", "sum"),
            lgas=("lga_code", "nunique"),
        )
        .reset_index()
        .sort_values("weight", ascending=False)
    )
    grouped["run_id"] = run_id
    return grouped


def _add_centrality(actors: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    try:
        import networkx as nx
    except ImportError:
        actors["degree_centrality"] = float("nan")
        return actors

    graph = nx.Graph()
    lga_edges = edges[edges["target_kind"] == "lga"]
    for row in lga_edges.itertuples():
        graph.add_edge(f"actor::{row.actor}", f"lga::{row.target}", weight=float(row.weight))
    centrality = nx.degree_centrality(graph)
    actors["degree_centrality"] = actors["actor"].map(
        lambda name: centrality.get(f"actor::{name}", 0.0)
    )
    return actors


def persist(graph: ThreatGraph) -> int:
    from ..storage import upsert

    if graph.edges.empty:
        return 0
    return upsert(
        "threat_actor_edges",
        graph.edges[
            [
                "run_id", "actor", "target_kind", "target", "events",
                "fatalities", "first_seen", "last_seen", "weight",
            ]
        ].to_dict(orient="records"),
    )
