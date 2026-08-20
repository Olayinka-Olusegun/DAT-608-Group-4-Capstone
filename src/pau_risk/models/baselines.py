"""Reference models that the hybrid has to beat to justify itself.

Each baseline stands for a way a state government could allocate escorts today
without any model at all, so the comparison measures the actual value added
rather than the distance from a strawman.

Persistence is what a commissioner does when they read last week's incident
report and cover the same places again.

Historical rate is the standing assumption that the dangerous LGAs are the ones
that have always been dangerous, implemented as the attack rate over the previous
year.

The Hawkes intensity on its own is included because it is a complete model in its
own right. If the gradient boosted layer cannot beat it, the extra machinery is
not earning its place, and saying so is more useful than hiding it.

Gradient boosting without the Hawkes features isolates the contribution of the
point process, which is the specific claim the architecture in the brief makes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HAWKES_PREFIX = "hawkes_"


def persistence(frame: pd.DataFrame) -> np.ndarray:
    """Score equals the number of attacks in the LGA last week."""
    if "own_events_1w" in frame:
        return frame["own_events_1w"].to_numpy(dtype=float)
    return np.zeros(len(frame))


def historical_rate(frame: pd.DataFrame) -> np.ndarray:
    """Score equals the attack count over the previous year."""
    column = "own_events_52w" if "own_events_52w" in frame else "hist_rate"
    return frame[column].to_numpy(dtype=float)


def hawkes_only(frame: pd.DataFrame) -> np.ndarray:
    """Score equals the probability of at least one event implied by the process."""
    if "hawkes_prob_any" in frame:
        return frame["hawkes_prob_any"].to_numpy(dtype=float)
    return np.zeros(len(frame))


def non_hawkes_features(features: list[str]) -> list[str]:
    return [name for name in features if not name.startswith(HAWKES_PREFIX)]
