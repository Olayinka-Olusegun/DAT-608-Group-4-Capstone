"""The explainability layer.

A risk tier that arrives without a reason cannot be acted on. If an LGA moves
from elevated to severe, the security council needs to know whether that is
because an attack happened next door on Tuesday, because a military operation
started in the adjacent LGA and pushed a group across the boundary, or because a
festival falls inside the window. Those three situations call for different
responses, and the number alone does not distinguish them.

SHAP values are used because they attribute a specific prediction to its inputs
additively and locally, so the contributions for one LGA in one week sum to the
distance between that LGA's score and the average score. Global feature
importance cannot do that: it describes the model, not the row. Each contribution
is then rendered with the human readable feature label and the underlying value,
so the panel reads as a sentence about that LGA rather than a column name.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.panel import FEATURE_LABELS
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class Explanation:
    drivers: pd.DataFrame          # long form, one row per LGA per driver
    global_importance: pd.DataFrame


def _explainer(booster):
    import shap

    return shap.TreeExplainer(booster)


def explain(
    booster,
    frame: pd.DataFrame,
    features: list[str],
    top_k: int = 5,
    key_columns: tuple[str, ...] = ("lga_code", "week_start"),
) -> Explanation:
    """Attribute each row's score to its features and keep the strongest drivers."""
    matrix = frame[features]
    shap_values = _explainer(booster).shap_values(matrix)
    if isinstance(shap_values, list):  # older SHAP returns one array per class
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values, dtype=float)

    order = np.argsort(-np.abs(shap_values), axis=1)[:, :top_k]
    values = matrix.to_numpy(dtype=float)

    records: list[dict] = []
    keys = frame[list(key_columns)].reset_index(drop=True)
    for position in range(len(frame)):
        base = {column: keys.iloc[position][column] for column in key_columns}
        for rank, column_index in enumerate(order[position], start=1):
            name = features[column_index]
            records.append(
                {
                    **base,
                    "driver_rank": rank,
                    "feature_name": name,
                    "feature_label": FEATURE_LABELS.get(name, name),
                    "feature_value": float(values[position, column_index]),
                    "shap_value": float(shap_values[position, column_index]),
                }
            )

    importance = (
        pd.DataFrame(
            {
                "feature": features,
                "label": [FEATURE_LABELS.get(name, name) for name in features],
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    LOGGER.info(
        "explained %d rows, strongest global driver is %s",
        len(frame),
        importance.iloc[0]["label"] if not importance.empty else "n/a",
    )
    return Explanation(drivers=pd.DataFrame.from_records(records), global_importance=importance)


def narrate(drivers: pd.DataFrame, lga_code: str) -> list[str]:
    """Render one LGA's drivers as plain statements for the brief and the dashboard."""
    subset = drivers[drivers["lga_code"] == lga_code].sort_values("driver_rank")
    lines: list[str] = []
    for row in subset.itertuples():
        direction = "raises" if row.shap_value > 0 else "lowers"
        value = row.feature_value
        rendered = f"{value:.2f}".rstrip("0").rstrip(".") if abs(value) < 1000 else f"{value:,.0f}"
        lines.append(f"{row.feature_label} at {rendered} {direction} the score")
    return lines
