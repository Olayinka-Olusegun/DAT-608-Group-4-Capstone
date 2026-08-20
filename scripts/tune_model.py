"""Small hyper-parameter sweep, selected on the validation fold only.

The test fold is not read here. Selection uses recall in the top twenty LGAs per
week on 2022, because that is the operational quantity the service is judged on,
and average precision is reported alongside it as a check that the improvement is
not confined to the very top of the ranking.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pau_risk.config import get_settings  # noqa: E402
from pau_risk.logging_utils import configure, get_logger  # noqa: E402
from pau_risk.models.metrics import evaluate  # noqa: E402
from pau_risk.models.train import temporal_split  # noqa: E402

LOGGER = get_logger("tune")

GRID = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.05],
    "scale_pos_weight": ["auto", 1.0],
    "min_child_weight": [5.0, 20.0],
}


def main() -> int:
    configure()
    settings = get_settings()
    model_cfg = settings.section("model")

    panel = pd.read_parquet(settings.paths.processed / "panel.parquet")
    coverage = pd.read_csv(settings.paths.artifacts / "feature_coverage.csv")
    features = coverage.loc[coverage["in_model"], "feature"].tolist()

    split = temporal_split(
        panel,
        pd.Timestamp(str(model_cfg["train_end"])).date(),
        pd.Timestamp(str(model_cfg["valid_end"])).date(),
    )
    positives = int(split.train["label"].sum())
    auto_weight = (len(split.train) - positives) / max(positives, 1)

    from xgboost import XGBClassifier

    rows = []
    keys = list(GRID)
    for combination in itertools.product(*(GRID[key] for key in keys)):
        parameters = dict(zip(keys, combination))
        weight = auto_weight if parameters["scale_pos_weight"] == "auto" else 1.0
        booster = XGBClassifier(
            n_estimators=int(model_cfg["n_estimators"]),
            learning_rate=parameters["learning_rate"],
            max_depth=parameters["max_depth"],
            min_child_weight=parameters["min_child_weight"],
            subsample=float(model_cfg["subsample"]),
            colsample_bytree=float(model_cfg["colsample_bytree"]),
            reg_lambda=float(model_cfg["reg_lambda"]),
            scale_pos_weight=weight,
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
        scored = split.valid[["week_start", "lga_code", "label"]].copy()
        scored["score"] = booster.predict_proba(split.valid[features])[:, 1]
        result = evaluate(scored, "candidate")
        rows.append(
            {
                **parameters,
                "best_iteration": int(getattr(booster, "best_iteration", 0)),
                "roc_auc": result.roc_auc,
                "average_precision": result.average_precision,
                "recall_at_20": result.at_k[20]["recall"],
                "precision_at_20": result.at_k[20]["precision"],
                "recall_at_50": result.at_k[50]["recall"],
            }
        )
        LOGGER.info(
            "depth=%s lr=%s weight=%s mcw=%s -> recall@20 %.3f, AP %.4f",
            parameters["max_depth"],
            parameters["learning_rate"],
            parameters["scale_pos_weight"],
            parameters["min_child_weight"],
            rows[-1]["recall_at_20"],
            rows[-1]["average_precision"],
        )

    table = pd.DataFrame(rows).sort_values(
        ["recall_at_20", "average_precision"], ascending=False
    )
    output = settings.paths.artifacts / "tuning_results.csv"
    table.to_csv(output, index=False)
    print(table.head(10).round(4).to_string(index=False))
    LOGGER.info("wrote %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
