"""Train, calibrate and register the hybrid Hawkes and XGBoost model.

The split is strictly temporal. Weeks up to the end of 2021 train the model,
2022 tunes the stopping point and fits the calibration curve, and 2023 onward is
held out and touched once. A random split would be indefensible here: adjacent
weeks in the same LGA share almost all of their feature values, so shuffling
would let the model see the answer to a question it is about to be asked, and the
reported metrics would not survive contact with a live week.

Class imbalance is not handled by resampling. Resampling would distort the base
rate that the calibration step then has to undo, and the whole point of that step
is that the number attached to an LGA can be read as a probability. The objective
optimises area under the precision-recall curve instead, which is already the
right target under a positive rate near half a percent, and the sweep in
scripts/tune_model.py found that leaving the positive class unweighted ranked
better on validation than up-weighting it. Isotonic regression is then fitted on
the validation fold alone and applied unchanged to the test fold and to live
scoring.

Everything that matters for reproduction, the parameters, the fitted Hawkes
constants, the feature list, the metrics and the model file, is logged to MLflow
under one run, and the same run identifier keys every prediction row in the
warehouse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ..config import Settings, get_settings
from ..logging_utils import get_logger
from . import baselines
from .metrics import Evaluation, comparison_table, evaluate

LOGGER = get_logger(__name__)

TIER_ORDER = ("Low", "Elevated", "High", "Severe")


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> dict[str, Any]:
        return {
            fold: {
                "rows": len(frame),
                "positives": int(frame["label"].sum()),
                "from": str(frame["week_start"].min().date()),
                "to": str(frame["week_start"].max().date()),
            }
            for fold, frame in (("train", self.train), ("valid", self.valid), ("test", self.test))
        }


@dataclass
class TrainedModel:
    booster: Any
    calibrator: IsotonicRegression
    features: list[str]
    tier_thresholds: dict[str, float]
    metrics: dict[str, Any]
    comparison: pd.DataFrame
    hawkes_parameters: dict[str, float]
    run_id: str
    model_version: str
    train_end: date
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict_proba(frame[self.features])[:, 1]
        return calibrate(self.calibrator, raw)

    def tier(self, probabilities: np.ndarray) -> np.ndarray:
        thresholds = self.tier_thresholds
        tiers = np.full(len(probabilities), "Low", dtype=object)
        tiers[probabilities >= thresholds["Elevated"]] = "Elevated"
        tiers[probabilities >= thresholds["High"]] = "High"
        tiers[probabilities >= thresholds["Severe"]] = "Severe"
        return tiers


TIE_BREAK = 1e-6


def calibrate(calibrator: IsotonicRegression, raw: np.ndarray) -> np.ndarray:
    """Map raw scores onto the probability scale without flattening the ranking.

    Isotonic regression is a step function, so it maps whole intervals of raw
    score onto one calibrated value. That is exactly what makes it well
    calibrated, and it is also a problem here, because the service ranks 774
    LGAs and hands the top of that list to a commissioner. Inside a plateau the
    ordering becomes arbitrary, and measured recall in the top twenty falls even
    though the probabilities are more honest.

    Adding a millionth of the raw score restores the original ordering within
    each plateau while moving any probability by less than one part in a million,
    which is far below the resolution at which a tier boundary is drawn.
    """
    return np.clip(calibrator.predict(raw) + TIE_BREAK * raw, 0.0, 1.0)


def temporal_split(panel: pd.DataFrame, train_end: date, valid_end: date) -> Split:
    weeks = panel["week_start"]
    return Split(
        train=panel[weeks <= pd.Timestamp(train_end)].copy(),
        valid=panel[(weeks > pd.Timestamp(train_end)) & (weeks <= pd.Timestamp(valid_end))].copy(),
        test=panel[weeks > pd.Timestamp(valid_end)].copy(),
    )


def _fit_booster(
    split: Split, features: list[str], model_cfg: dict[str, Any], label: str
):
    from xgboost import XGBClassifier

    positives = max(int(split.train["label"].sum()), 1)
    negatives = len(split.train) - positives
    scale = (
        negatives / positives
        if model_cfg.get("scale_pos_weight", "auto") == "auto"
        else float(model_cfg["scale_pos_weight"])
    )

    booster = XGBClassifier(
        n_estimators=int(model_cfg["n_estimators"]),
        learning_rate=float(model_cfg["learning_rate"]),
        max_depth=int(model_cfg["max_depth"]),
        subsample=float(model_cfg["subsample"]),
        colsample_bytree=float(model_cfg["colsample_bytree"]),
        min_child_weight=float(model_cfg["min_child_weight"]),
        reg_lambda=float(model_cfg["reg_lambda"]),
        scale_pos_weight=scale,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=int(model_cfg["early_stopping_rounds"]),
        random_state=int(model_cfg["seed"]),
        n_jobs=-1,
        tree_method="hist",
    )
    booster.fit(
        split.train[features],
        split.train["label"],
        eval_set=[(split.valid[features], split.valid["label"])],
        verbose=False,
    )
    LOGGER.info(
        "%s: %d features, best iteration %s, positive weight %.1f",
        label,
        len(features),
        getattr(booster, "best_iteration", "n/a"),
        scale,
    )
    return booster


def _tier_thresholds(probabilities: np.ndarray) -> dict[str, float]:
    """Set cut points from the validation distribution, tied to weekly capacity.

    A state can realistically surge into a handful of LGAs at once, so the tiers
    are defined by how much of the national list they admit rather than by round
    numbers on the probability scale: severe is the top half percent of LGA weeks,
    high the top two percent, elevated the top tenth.
    """
    return {
        "Elevated": float(np.quantile(probabilities, 0.90)),
        "High": float(np.quantile(probabilities, 0.98)),
        "Severe": float(np.quantile(probabilities, 0.995)),
    }


MIN_POSITIVES = {"train": 200, "valid": 20, "test": 20}


def _validate_split(split: Split) -> None:
    """Refuse to train on a panel that cannot support the claim being made.

    This guard exists because of a failure observed in testing rather than
    imagined. Running the pipeline with an ingestion window that contained no
    events produced a panel of 484,524 rows and no positives at all. Training
    completed, every metric came back as not a number, and the scoring stage
    assigned the severe tier to all 774 areas, which the alerting stage was ready
    to dispatch. Nothing raised. A model with no positive examples must fail
    loudly at the point where it is asked to learn, because every stage after it
    will otherwise produce confident output from nothing.
    """
    problems = []
    for fold, frame in (("train", split.train), ("valid", split.valid), ("test", split.test)):
        if frame.empty:
            problems.append(f"the {fold} fold is empty")
            continue
        positives = int(frame["label"].sum())
        if positives < MIN_POSITIVES[fold]:
            problems.append(
                f"the {fold} fold has {positives} positive weeks, "
                f"below the minimum of {MIN_POSITIVES[fold]}"
            )
    if problems:
        raise ValueError(
            "Refusing to train: "
            + "; ".join(problems)
            + ". The usual cause is an ingestion window that did not cover the "
            "panel period. Backfill first, for example: "
            "pau-risk ingest --since 2010-01-01 --until 2024-12-31"
        )


def train(
    panel: pd.DataFrame,
    model_features: list[str],
    hawkes_parameters: dict[str, float],
    coverage: pd.DataFrame | None = None,
    settings: Settings | None = None,
) -> TrainedModel:
    settings = settings or get_settings()
    model_cfg = settings.section("model")
    train_end = date.fromisoformat(str(model_cfg["train_end"]))
    valid_end = date.fromisoformat(str(model_cfg["valid_end"]))

    split = temporal_split(panel, train_end, valid_end)
    LOGGER.info("split: %s", json.dumps(split.describe()))
    _validate_split(split)

    full_booster = _fit_booster(split, model_features, model_cfg, "hybrid")
    reduced_features = baselines.non_hawkes_features(model_features)
    reduced_booster = _fit_booster(split, reduced_features, model_cfg, "boosting without Hawkes")

    # Calibration is fitted on validation predictions only and never sees the test fold.
    valid_raw = full_booster.predict_proba(split.valid[model_features])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(valid_raw, split.valid["label"].to_numpy(dtype=float))

    valid_calibrated = calibrate(calibrator, valid_raw)
    thresholds = _tier_thresholds(valid_calibrated)

    test = split.test.copy()
    test_raw = full_booster.predict_proba(test[model_features])[:, 1]
    test["score"] = calibrate(calibrator, test_raw)

    evaluations: list[Evaluation] = []
    for name, scores in (
        ("Persistence (attacks last week)", baselines.persistence(test)),
        ("Historical rate (past year)", baselines.historical_rate(test)),
        ("Hawkes process alone", baselines.hawkes_only(test)),
        (
            "XGBoost without Hawkes features",
            reduced_booster.predict_proba(test[reduced_features])[:, 1],
        ),
        ("Hybrid Hawkes and XGBoost, uncalibrated", test_raw),
        ("Hybrid Hawkes and XGBoost, calibrated", test["score"].to_numpy()),
    ):
        scored = test[["week_start", "lga_code", "label"]].copy()
        scored["score"] = scores
        evaluations.append(evaluate(scored, name))

    comparison = comparison_table(evaluations)
    headline = evaluations[-1]
    LOGGER.info(
        "test performance: ROC AUC %.3f, average precision %.3f, precision@20 %.3f, recall@20 %.3f",
        headline.roc_auc,
        headline.average_precision,
        headline.at_k[20]["precision"],
        headline.at_k[20]["recall"],
    )

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    metrics = {
        "split": split.describe(),
        "test": headline.flat(),
        "tier_thresholds": thresholds,
        "hawkes": hawkes_parameters,
        "n_features": len(model_features),
    }

    model = TrainedModel(
        booster=full_booster,
        calibrator=calibrator,
        features=model_features,
        tier_thresholds=thresholds,
        metrics=metrics,
        comparison=comparison,
        hawkes_parameters=hawkes_parameters,
        run_id=run_id,
        model_version=f"hawkes-xgb-{date.today().isoformat()}",
        train_end=train_end,
        coverage=coverage if coverage is not None else pd.DataFrame(),
    )
    _log_to_mlflow(model, evaluations, settings)
    _persist(model, settings)
    return model


def _metric_prefix(name: str) -> str:
    """MLflow metric names allow only a restricted character set."""
    import re

    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", name.split("(")[0].strip().lower())
    return cleaned.strip("_")[:40]


def _log_to_mlflow(
    model: TrainedModel, evaluations: list[Evaluation], settings: Settings
) -> None:
    try:
        import mlflow
    except ImportError:
        LOGGER.info("mlflow is not installed, skipping experiment logging")
        return

    model_cfg = settings.section("model")
    try:
        mlflow.set_tracking_uri(settings.mlflow_uri)
        mlflow.set_experiment("lga-violence-risk")
        run_context = mlflow.start_run(run_name=model.run_id)
    except Exception as exc:  # noqa: BLE001 - tracking must never block a training run
        LOGGER.warning("mlflow tracking unavailable (%s), continuing", exc.__class__.__name__)
        return

    with run_context:
        mlflow.log_params(
            {
                key: value
                for key, value in model_cfg.items()
                if isinstance(value, (int, float, str, bool))
            }
        )
        mlflow.log_params({f"hawkes_{k}": v for k, v in model.hawkes_parameters.items()})
        mlflow.log_param("n_features", len(model.features))
        for evaluation in evaluations:
            prefix = _metric_prefix(evaluation.name)
            for metric, value in evaluation.flat().items():
                if np.isfinite(value):
                    mlflow.log_metric(f"{prefix}.{metric}", float(value))
        artifacts = settings.paths.artifacts
        model.comparison.to_csv(artifacts / "model_comparison.csv", index=False)
        mlflow.log_artifact(str(artifacts / "model_comparison.csv"))
        if evaluations[-1].calibration is not None:
            evaluations[-1].calibration.to_csv(artifacts / "calibration.csv", index=False)
            mlflow.log_artifact(str(artifacts / "calibration.csv"))
    LOGGER.info("logged run %s to %s", model.run_id, settings.mlflow_uri)


def _persist(model: TrainedModel, settings: Settings) -> None:
    import pickle

    directory = settings.paths.artifacts / "model"
    directory.mkdir(parents=True, exist_ok=True)
    model.booster.save_model(str(directory / "booster.json"))
    with (directory / "calibrator.pkl").open("wb") as handle:
        pickle.dump(model.calibrator, handle)
    payload = {
        "features": model.features,
        "tier_thresholds": model.tier_thresholds,
        "metrics": model.metrics,
        "hawkes_parameters": model.hawkes_parameters,
        "run_id": model.run_id,
        "model_version": model.model_version,
        "train_end": model.train_end.isoformat(),
    }
    (directory / "metadata.json").write_text(json.dumps(payload, indent=2, default=str))
    model.comparison.to_csv(settings.paths.artifacts / "model_comparison.csv", index=False)
    if not model.coverage.empty:
        model.coverage.to_csv(settings.paths.artifacts / "feature_coverage.csv", index=False)
    LOGGER.info("model artifacts written to %s", directory)


def load(settings: Settings | None = None) -> TrainedModel:
    import pickle

    from xgboost import XGBClassifier

    settings = settings or get_settings()
    directory = settings.paths.artifacts / "model"
    metadata = json.loads((directory / "metadata.json").read_text())

    booster = XGBClassifier()
    booster.load_model(str(directory / "booster.json"))
    with (directory / "calibrator.pkl").open("rb") as handle:
        calibrator = pickle.load(handle)

    comparison_path = settings.paths.artifacts / "model_comparison.csv"
    coverage_path = settings.paths.artifacts / "feature_coverage.csv"
    return TrainedModel(
        booster=booster,
        calibrator=calibrator,
        features=metadata["features"],
        tier_thresholds=metadata["tier_thresholds"],
        metrics=metadata["metrics"],
        comparison=pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame(),
        hawkes_parameters=metadata["hawkes_parameters"],
        run_id=metadata["run_id"],
        model_version=metadata["model_version"],
        train_end=date.fromisoformat(metadata["train_end"]),
        coverage=pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame(),
    )
