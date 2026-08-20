"""Evaluation built around the decision the score actually supports.

Area under the ROC curve is reported because it is expected, but it is close to
useless on its own here. With a positive rate near half a percent, a model can
post a strong ROC AUC while still filling the top of the list with false alarms.
The question a commissioner of police asks is narrower and harder: if I can cover
twenty local government areas this week, how many of the areas that were actually
attacked did the list contain.

Precision and recall at k answer that directly, computed within each week rather
than pooled, because the ranking is used one week at a time. Average precision
summarises the whole ranking under the same imbalance. The Brier score and the
calibration table check the other half of the requirement in the brief, which is
that the number reported is a probability and not just an ordering: a tier called
severe has to mean something stable about how often an attack follows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass
class Evaluation:
    name: str
    roc_auc: float
    average_precision: float
    brier: float
    positives: int
    rows: int
    at_k: dict[int, dict[str, float]] = field(default_factory=dict)
    calibration: pd.DataFrame | None = None

    def flat(self) -> dict[str, float]:
        values: dict[str, float] = {
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "brier": self.brier,
            "positives": float(self.positives),
            "rows": float(self.rows),
        }
        for k, scores in self.at_k.items():
            for metric, value in scores.items():
                values[f"{metric}_at_{k}"] = value
        return values

    def summary_row(self) -> dict[str, float | str]:
        row: dict[str, float | str] = {"model": self.name}
        row.update(self.flat())
        return row


def precision_recall_at_k(
    frame: pd.DataFrame, k: int, score_column: str = "score", label_column: str = "label"
) -> dict[str, float]:
    """Average per-week precision, recall and lift over the top k ranked LGAs."""
    precisions, recalls, lifts = [], [], []
    for _, week in frame.groupby("week_start"):
        if week.empty:
            continue
        positives = int(week[label_column].sum())
        top = week.nlargest(min(k, len(week)), score_column)
        hits = int(top[label_column].sum())
        precision = hits / len(top)
        precisions.append(precision)
        recalls.append(hits / positives if positives else np.nan)
        base_rate = positives / len(week)
        lifts.append(precision / base_rate if base_rate > 0 else np.nan)
    return {
        "precision": float(np.nanmean(precisions)) if precisions else float("nan"),
        "recall": float(np.nanmean(recalls)) if recalls else float("nan"),
        "lift": float(np.nanmean(lifts)) if lifts else float("nan"),
    }


def calibration_table(
    labels: np.ndarray, scores: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    """Observed versus predicted rate by score decile."""
    frame = pd.DataFrame({"label": labels, "score": scores})
    frame["bucket"] = pd.qcut(frame["score"].rank(method="first"), bins, labels=False)
    grouped = frame.groupby("bucket").agg(
        rows=("label", "size"),
        predicted=("score", "mean"),
        observed=("label", "mean"),
    )
    grouped["gap"] = grouped["observed"] - grouped["predicted"]
    return grouped.reset_index()


def evaluate(
    frame: pd.DataFrame,
    name: str,
    score_column: str = "score",
    label_column: str = "label",
    k_values: tuple[int, ...] = (10, 20, 50),
) -> Evaluation:
    labels = frame[label_column].to_numpy(dtype=int)
    scores = frame[score_column].to_numpy(dtype=float)
    positives = int(labels.sum())

    roc = roc_auc_score(labels, scores) if 0 < positives < len(labels) else float("nan")
    ap = average_precision_score(labels, scores) if positives else float("nan")
    clipped = np.clip(scores, 0.0, 1.0)
    brier = brier_score_loss(labels, clipped) if positives else float("nan")

    return Evaluation(
        name=name,
        roc_auc=float(roc),
        average_precision=float(ap),
        brier=float(brier),
        positives=positives,
        rows=len(frame),
        at_k={k: precision_recall_at_k(frame, k, score_column, label_column) for k in k_values},
        calibration=calibration_table(labels, clipped) if positives else None,
    )


def comparison_table(evaluations: list[Evaluation]) -> pd.DataFrame:
    return pd.DataFrame([evaluation.summary_row() for evaluation in evaluations])
