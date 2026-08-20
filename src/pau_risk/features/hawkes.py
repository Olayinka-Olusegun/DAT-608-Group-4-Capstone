"""A multivariate Hawkes process over the LGA adjacency graph.

Why a Hawkes process at all. Armed group violence is not a sequence of
independent draws. An attack raises the chance of another attack in the same
place for a while afterwards, because the group is present, the route worked and
the response has not yet arrived, and it raises the chance in neighbouring areas
because that is where the group moves next. A gradient boosted tree given only
lagged counts has to learn that decay shape from data, using a separate parameter
for every lag window. A Hawkes process states it directly as a self exciting
point process and estimates three quantities that mean something operationally:
how much of the violence is offspring of earlier violence, how that offspring
splits between the same LGA and its neighbours, and how quickly the excitation
decays. Those estimates then enter the tree as features, which is the hybrid the
brief specifies.

The model is

    lambda_i(t) = mu_i
                + alpha_self  * sum over past events in i of  beta * exp(-beta * (t - s))
                + alpha_nb    * sum over neighbours j, weighted by w_ij, of the same term

with the kernel normalised so that alpha is the expected number of direct
offspring. The baseline mu_i is anchored to each LGA's own historical rate and
scaled by a single fitted parameter, which keeps the parameter count at four
across 774 dimensions instead of 777 and stops the baseline absorbing the
excitation it is supposed to separate out.

The two excitation parameters are reparameterised as a branching ratio rho and a
self share phi, so that alpha_self = rho * phi and alpha_nb = rho * (1 - phi).
Because the neighbour weights are row normalised, every row of the branching
matrix sums to rho, so stationarity is exactly rho < 1 and is enforced by
construction rather than by a penalty. It also makes the fitted values directly
readable: rho is the share of events that are triggered by an earlier event, and
phi says how much of that contagion stays inside the LGA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

MAX_BRANCHING = 0.95     # keeps the fitted process stationary
EPSILON = 1e-12


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _logit(value: float) -> float:
    value = min(max(value, 1e-6), 1 - 1e-6)
    return math.log(value / (1 - value))


def decay_sums(
    evaluation_times: np.ndarray, source_times: np.ndarray, beta: float
) -> np.ndarray:
    """Sum of exp(-beta * (t - s)) over source events strictly before each t.

    Evaluated by forward recursion rather than by the closed form, because the
    closed form factorises into exp(beta * s) and overflows once the observation
    window is a few thousand days long at realistic decay rates.
    """
    out = np.zeros(len(evaluation_times), dtype=float)
    if len(source_times) == 0 or len(evaluation_times) == 0:
        return out

    running = 0.0
    previous_t = evaluation_times[0]
    cursor = 0
    n_sources = len(source_times)

    for index, t in enumerate(evaluation_times):
        if index > 0:
            running *= math.exp(-beta * (t - previous_t))
        while cursor < n_sources and source_times[cursor] < t:
            running += math.exp(-beta * (t - source_times[cursor]))
            cursor += 1
        out[index] = running
        previous_t = t
    return out


def compensator_terms(
    source_times: np.ndarray, horizon: float, beta: float
) -> float:
    """Integral over the observation window of the kernel triggered by past events."""
    if len(source_times) == 0:
        return 0.0
    return float(np.sum(1.0 - np.exp(-beta * (horizon - source_times))))


@dataclass
class HawkesParameters:
    mu_scale: float
    branching_ratio: float
    self_share: float
    decay: float
    log_likelihood: float = float("nan")
    n_events: int = 0
    n_dimensions: int = 0

    @property
    def alpha_self(self) -> float:
        return self.branching_ratio * self.self_share

    @property
    def alpha_neighbour(self) -> float:
        return self.branching_ratio * (1.0 - self.self_share)

    @property
    def half_life_days(self) -> float:
        return math.log(2.0) / self.decay if self.decay > 0 else float("inf")

    def as_dict(self) -> dict[str, float]:
        return {
            "mu_scale": self.mu_scale,
            "branching_ratio": self.branching_ratio,
            "self_share": self.self_share,
            "decay_per_day": self.decay,
            "half_life_days": self.half_life_days,
            "alpha_self": self.alpha_self,
            "alpha_neighbour": self.alpha_neighbour,
            "log_likelihood": self.log_likelihood,
            "n_events": self.n_events,
            "n_dimensions": self.n_dimensions,
        }


@dataclass
class HawkesModel:
    """Fit and evaluate the process over a fixed adjacency structure."""

    neighbours: Mapping[str, list[tuple[str, float]]]
    codes: Sequence[str]
    parameters: HawkesParameters | None = None
    _events: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _baseline: dict[str, float] = field(default_factory=dict, init=False)
    _origin: pd.Timestamp | None = field(default=None, init=False)
    _horizon: float = field(default=0.0, init=False)

    # ------------------------------------------------------------------ data
    def load_events(
        self, events: pd.DataFrame, origin: pd.Timestamp, horizon_end: pd.Timestamp
    ) -> None:
        """Index event times per LGA as days since the origin."""
        self._origin = origin
        self._horizon = float((horizon_end - origin).days)
        times = (events["event_date"] - origin).dt.days.astype(float)
        frame = pd.DataFrame({"lga_code": events["lga_code"].to_numpy(), "t": times.to_numpy()})
        frame = frame[(frame["t"] >= 0) & (frame["t"] <= self._horizon)]
        grouped = frame.sort_values("t").groupby("lga_code")["t"]
        self._events = {code: series.to_numpy(dtype=float) for code, series in grouped}

        total = float(sum(len(values) for values in self._events.values()))
        # Laplace smoothed empirical rate per day, the anchor for each baseline.
        for code in self.codes:
            count = len(self._events.get(code, ()))
            self._baseline[code] = (count + 0.5) / max(self._horizon, 1.0)
        LOGGER.info(
            "hawkes corpus: %d events across %d active LGAs over %.0f days",
            int(total),
            len(self._events),
            self._horizon,
        )

    # ------------------------------------------------------------ likelihood
    def _negative_log_likelihood(self, theta: np.ndarray) -> float:
        mu_scale = math.exp(theta[0])
        rho = MAX_BRANCHING * _sigmoid(theta[1])
        phi = _sigmoid(theta[2])
        beta = math.exp(theta[3])

        alpha_self = rho * phi
        alpha_nb = rho * (1.0 - phi)

        log_term = 0.0
        compensator = 0.0

        for code in self.codes:
            own = self._events.get(code)
            mu = mu_scale * self._baseline[code]
            compensator += mu * self._horizon

            neighbour_pairs = self.neighbours.get(code, [])
            if own is not None and len(own):
                intensity = np.full(len(own), mu, dtype=float)
                intensity += alpha_self * beta * decay_sums(own, own, beta)
                for other, weight in neighbour_pairs:
                    source = self._events.get(other)
                    if source is None or not len(source):
                        continue
                    intensity += alpha_nb * weight * beta * decay_sums(own, source, beta)
                log_term += float(np.sum(np.log(np.maximum(intensity, EPSILON))))

                compensator += alpha_self * compensator_terms(own, self._horizon, beta)
            for other, weight in neighbour_pairs:
                source = self._events.get(other)
                if source is None or not len(source):
                    continue
                compensator += alpha_nb * weight * compensator_terms(source, self._horizon, beta)

        return -(log_term - compensator)

    def fit(self, decay_init: float = 0.05, max_iterations: int = 200) -> HawkesParameters:
        theta0 = np.array(
            [math.log(1.0), _logit(0.4 / MAX_BRANCHING), _logit(0.6), math.log(decay_init)]
        )
        result = minimize(
            self._negative_log_likelihood,
            theta0,
            method="L-BFGS-B",
            bounds=[(-6.0, 6.0), (-8.0, 8.0), (-8.0, 8.0), (math.log(1e-3), math.log(2.0))],
            options={"maxiter": max_iterations, "ftol": 1e-8},
        )
        theta = result.x
        self.parameters = HawkesParameters(
            mu_scale=math.exp(theta[0]),
            branching_ratio=MAX_BRANCHING * _sigmoid(theta[1]),
            self_share=_sigmoid(theta[2]),
            decay=math.exp(theta[3]),
            log_likelihood=-float(result.fun),
            n_events=int(sum(len(values) for values in self._events.values())),
            n_dimensions=len(self.codes),
        )
        LOGGER.info(
            "hawkes fit: branching=%.3f self_share=%.3f half_life=%.1f days logL=%.1f",
            self.parameters.branching_ratio,
            self.parameters.self_share,
            self.parameters.half_life_days,
            self.parameters.log_likelihood,
        )
        return self.parameters

    # -------------------------------------------------------------- features
    def intensity_frame(
        self, evaluation_dates: Iterable[pd.Timestamp], horizon_days: int = 7
    ) -> pd.DataFrame:
        """Decompose the conditional intensity at each evaluation date per LGA.

        Only events strictly before an evaluation date contribute, which is what
        makes these safe to use as features for that date. The expected count
        over the horizon integrates the current kernel forward and deliberately
        ignores offspring of events that have not happened yet, so it is a lower
        bound rather than a simulation.
        """
        if self.parameters is None:
            raise RuntimeError("fit the process before generating features")
        if self._origin is None:
            raise RuntimeError("load events before generating features")

        parameters = self.parameters
        beta = parameters.decay
        alpha_self = parameters.alpha_self
        alpha_nb = parameters.alpha_neighbour

        stamps = pd.DatetimeIndex(sorted(set(evaluation_dates)))
        grid = np.array([(stamp - self._origin).days for stamp in stamps], dtype=float)
        decay_integral = (1.0 - math.exp(-beta * horizon_days)) / beta

        rows: list[dict] = []
        for code in self.codes:
            mu = parameters.mu_scale * self._baseline[code]
            own = self._events.get(code, np.empty(0))
            self_r = decay_sums(grid, own, beta)
            neighbour_r = np.zeros_like(grid)
            for other, weight in self.neighbours.get(code, []):
                source = self._events.get(other)
                if source is None or not len(source):
                    continue
                neighbour_r += weight * decay_sums(grid, source, beta)

            self_term = alpha_self * beta * self_r
            neighbour_term = alpha_nb * beta * neighbour_r
            total = mu + self_term + neighbour_term
            expected = mu * horizon_days + (alpha_self * self_r + alpha_nb * neighbour_r) * (
                beta * decay_integral
            )

            for index, stamp in enumerate(stamps):
                rows.append(
                    {
                        "lga_code": code,
                        "week_start": stamp,
                        "hawkes_background": mu,
                        "hawkes_self": float(self_term[index]),
                        "hawkes_neighbour": float(neighbour_term[index]),
                        "hawkes_total": float(total[index]),
                        "hawkes_self_share": float(self_term[index] / max(total[index], EPSILON)),
                        "hawkes_neighbour_share": float(
                            neighbour_term[index] / max(total[index], EPSILON)
                        ),
                        "hawkes_expected_events": float(expected[index]),
                        "hawkes_prob_any": float(1.0 - math.exp(-max(expected[index], 0.0))),
                    }
                )
        frame = pd.DataFrame.from_records(rows)
        LOGGER.info("hawkes features: %d rows over %d dates", len(frame), len(stamps))
        return frame


def neighbours_from_adjacency(adjacency: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """Group the adjacency table into the mapping the process expects."""
    mapping: dict[str, list[tuple[str, float]]] = {}
    for row in adjacency.itertuples():
        mapping.setdefault(row.lga_code, []).append((row.neighbour_code, float(row.weight)))
    return mapping
