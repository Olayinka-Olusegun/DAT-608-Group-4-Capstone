"""The point process is the part of the model most likely to be quietly wrong.

Two properties are checked. The recursion that evaluates the excitation sums has
to agree with the direct double sum it replaces, because it is the only place in
the model where an ordering mistake would produce plausible but incorrect
numbers. And the estimator has to recover parameters it generated itself, which
is the only way to tell that the likelihood and the simulation agree about what
the model means.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pau_risk.features.hawkes import (
    HawkesModel,
    decay_sums,
    neighbours_from_adjacency,
)


def brute_force(evaluation_times, source_times, beta):
    return np.array(
        [
            sum(math.exp(-beta * (t - s)) for s in source_times if s < t)
            for t in evaluation_times
        ]
    )


def test_decay_recursion_matches_direct_summation():
    rng = np.random.default_rng(608)
    source = np.sort(rng.uniform(0, 500, size=120))
    evaluation = np.sort(rng.uniform(0, 500, size=80))
    for beta in (0.01, 0.05, 0.3):
        assert np.allclose(
            decay_sums(evaluation, source, beta), brute_force(evaluation, source, beta), atol=1e-9
        )


def test_decay_excludes_simultaneous_events():
    """An event must not excite itself or anything sharing its timestamp."""
    times = np.array([10.0, 10.0, 20.0])
    result = decay_sums(times, times, beta=0.1)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(2 * math.exp(-1.0))


def test_empty_history_gives_zero_excitation():
    assert np.all(decay_sums(np.array([1.0, 2.0]), np.empty(0), 0.1) == 0.0)


def _simulate(codes, neighbours, mu, alpha_self, alpha_nb, beta, horizon, seed=3):
    """Ogata thinning over all dimensions jointly.

    The dimensions must be advanced together on one clock. Simulating each LGA to
    the horizon in turn would leave the first LGA with no neighbour history to be
    excited by, which silently removes the cross term the estimator is being
    asked to recover.
    """
    rng = np.random.default_rng(seed)
    index = {code: position for position, code in enumerate(codes)}
    neighbour_index = [
        [(index[other], weight) for other, weight in neighbours.get(code, [])] for code in codes
    ]
    events: list[list[float]] = [[] for _ in codes]

    def intensities(t: float) -> np.ndarray:
        values = np.full(len(codes), mu, dtype=float)
        for position in range(len(codes)):
            own = np.asarray(events[position])
            if own.size:
                values[position] += alpha_self * beta * float(np.sum(np.exp(-beta * (t - own))))
            for other, weight in neighbour_index[position]:
                history = np.asarray(events[other])
                if history.size:
                    values[position] += (
                        alpha_nb * weight * beta * float(np.sum(np.exp(-beta * (t - history))))
                    )
        return values

    t = 0.0
    while t < horizon:
        bound = float(intensities(t).sum())
        if bound <= 0:
            break
        t += rng.exponential(1.0 / bound)
        if t >= horizon:
            break
        current = intensities(t)
        total = float(current.sum())
        if rng.uniform() < total / bound:
            events[int(rng.choice(len(codes), p=current / total))].append(t)
    return {code: events[position] for position, code in enumerate(codes)}


def test_estimator_recovers_simulated_parameters():
    codes = [f"L{i:02d}" for i in range(12)]
    adjacency = pd.DataFrame(
        [
            {"lga_code": codes[i], "neighbour_code": codes[(i + 1) % len(codes)], "weight": 0.5},
            {"lga_code": codes[i], "neighbour_code": codes[(i - 1) % len(codes)], "weight": 0.5},
        ][j]
        for i in range(len(codes))
        for j in (0, 1)
    )
    neighbours = neighbours_from_adjacency(adjacency)

    true_mu, true_alpha_self, true_alpha_nb, true_beta = 0.02, 0.35, 0.10, 0.05
    horizon = 4000.0
    simulated = _simulate(codes, neighbours, true_mu, true_alpha_self, true_alpha_nb, true_beta, horizon)

    origin = pd.Timestamp("2015-01-05")
    rows = [
        {"lga_code": code, "event_date": origin + pd.Timedelta(days=float(t))}
        for code, times in simulated.items()
        for t in times
    ]
    frame = pd.DataFrame(rows)
    assert len(frame) > 200, "simulation produced too few events to identify the parameters"

    model = HawkesModel(neighbours=neighbours, codes=codes)
    model.load_events(frame, origin=origin, horizon_end=origin + pd.Timedelta(days=horizon))
    fitted = model.fit(decay_init=0.08, max_iterations=300)

    # The realised event count is itself a check on the simulation: a branching
    # ratio of rho inflates the background count by a factor of one over one
    # minus rho, so a badly specified simulator shows up here before the fit does.
    expected_events = len(codes) * true_mu * horizon / (1 - true_alpha_self - true_alpha_nb)
    assert len(frame) == pytest.approx(expected_events, rel=0.25)

    true_branching = true_alpha_self + true_alpha_nb
    assert fitted.branching_ratio == pytest.approx(true_branching, abs=0.15)
    assert fitted.self_share == pytest.approx(true_alpha_self / true_branching, abs=0.15)
    assert fitted.decay == pytest.approx(true_beta, rel=0.5)


def test_intensity_features_are_causal():
    """Intensity at a date must ignore events on or after that date."""
    codes = ["A", "B"]
    adjacency = pd.DataFrame(
        [{"lga_code": "A", "neighbour_code": "B", "weight": 1.0},
         {"lga_code": "B", "neighbour_code": "A", "weight": 1.0}]
    )
    origin = pd.Timestamp("2020-01-06")
    events = pd.DataFrame(
        {
            "lga_code": ["A", "A", "B"],
            "event_date": [
                origin + pd.Timedelta(days=10),
                origin + pd.Timedelta(days=200),
                origin + pd.Timedelta(days=15),
            ],
        }
    )
    model = HawkesModel(neighbours=neighbours_from_adjacency(adjacency), codes=codes)
    model.load_events(events, origin=origin, horizon_end=origin + pd.Timedelta(days=365))
    model.fit(max_iterations=50)

    before = model.intensity_frame([origin + pd.Timedelta(days=5)])
    row = before[before["lga_code"] == "A"].iloc[0]
    assert row["hawkes_self"] == pytest.approx(0.0)
    assert row["hawkes_neighbour"] == pytest.approx(0.0)

    after = model.intensity_frame([origin + pd.Timedelta(days=20)])
    row = after[after["lga_code"] == "A"].iloc[0]
    assert row["hawkes_self"] > 0
    assert row["hawkes_neighbour"] > 0
