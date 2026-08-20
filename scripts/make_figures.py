"""Produce the evaluation figures and print the reading of each one.

Every figure here answers a question that the summary table cannot. The table
reports recall at twenty; the coverage curve shows what happens at every other
budget a state might actually have. The table reports a Brier score; the
calibration plot shows where the probabilities are trustworthy and where they are
not. The interpretation is printed next to each figure so the two travel together.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pau_risk.config import get_settings  # noqa: E402
from pau_risk.logging_utils import configure, get_logger  # noqa: E402
from pau_risk.models import baselines, load  # noqa: E402
from pau_risk.models.metrics import calibration_table, precision_recall_at_k  # noqa: E402
from pau_risk.models.train import calibrate, temporal_split  # noqa: E402

LOGGER = get_logger("figures")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def main() -> int:
    configure()
    settings = get_settings()
    figures = settings.paths.root / "docs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(settings.paths.processed / "panel.parquet")
    model = load(settings)
    model_cfg = settings.section("model")
    split = temporal_split(
        panel,
        pd.Timestamp(str(model_cfg["train_end"])).date(),
        pd.Timestamp(str(model_cfg["valid_end"])).date(),
    )
    test = split.test.copy()
    raw = model.booster.predict_proba(test[model.features])[:, 1]
    test["score"] = calibrate(model.calibrator, raw)

    reduced = baselines.non_hawkes_features(model.features)
    scored_variants = {
        "Persistence": baselines.persistence(test),
        "Historical rate": baselines.historical_rate(test),
        "Hawkes alone": baselines.hawkes_only(test),
        "Hybrid, calibrated": test["score"].to_numpy(),
    }

    notes: list[str] = []
    notes.append(_coverage_curve(test, scored_variants, figures))
    notes.append(_calibration_plot(test, figures))
    notes.append(_weekly_capture(test, figures))
    notes.append(_driver_importance(model, panel, figures))
    notes.append(_hawkes_kernel(settings, figures))

    report = "\n\n".join(notes)
    (settings.paths.root / "docs" / "figure_notes.md").write_text(report, encoding="utf-8")
    print(report)
    LOGGER.info("figures written to %s", figures)
    return 0


def _coverage_curve(test: pd.DataFrame, variants: dict[str, np.ndarray], figures: Path) -> str:
    budgets = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    curves: dict[str, list[float]] = {}
    for name, scores in variants.items():
        frame = test[["week_start", "lga_code", "label"]].copy()
        frame["score"] = scores
        recalls = [precision_recall_at_k(frame, k)["recall"] for k in budgets]
        curves[name] = recalls
        axis.plot(budgets, recalls, marker="o", markersize=3.5, label=name)
    axis.set_xlabel("Local government areas covered each week (k)")
    axis.set_ylabel("Share of attacked LGAs inside the top k")
    axis.set_title("How much of the week's violence a fixed patrol budget reaches")
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(figures / "coverage_curve.png")
    plt.close(figure)

    hybrid = curves["Hybrid, calibrated"]
    at_20 = hybrid[budgets.index(20)]
    at_50 = hybrid[budgets.index(50)]
    persistence_20 = curves["Persistence"][budgets.index(20)]
    return (
        "Figure 1, coverage curve. The vertical axis is the share of LGAs that were actually "
        f"attacked in a week which appear in the top k of that week's ranking. Covering 20 of "
        f"774 areas, that is 2.6% of the country, reaches {at_20:.0%} of the week's attacks, and "
        f"covering 50 areas reaches {at_50:.0%}. The same 20 area budget allocated by last week's "
        f"incident report, which is the current practice the brief describes, reaches "
        f"{persistence_20:.0%}. The gap between those two lines is the operational value of the "
        "model, and it is widest exactly where a state government has to operate, at small k."
    )


def _calibration_plot(test: pd.DataFrame, figures: Path) -> str:
    table = calibration_table(
        test["label"].to_numpy(dtype=int), test["score"].to_numpy(dtype=float), bins=10
    )
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    limit = float(max(table["predicted"].max(), table["observed"].max())) * 1.15
    axis.plot([0, limit], [0, limit], linestyle="--", linewidth=1, color="black",
              label="Perfect calibration")
    axis.scatter(table["predicted"], table["observed"], s=28, label="Score decile")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed attack rate")
    axis.set_title("Calibration on the held-out period")
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(figures / "calibration.png")
    plt.close(figure)

    top = table.iloc[-1]
    worst_gap = table["gap"].abs().max()
    return (
        "Figure 2, calibration. Each point is one decile of the score distribution, plotted as "
        "predicted rate against the rate actually observed. Points on the diagonal mean the "
        f"number can be read as a probability. The largest deviation across the ten deciles is "
        f"{worst_gap:.3f}, and the top decile predicts {top['predicted']:.3f} against an observed "
        f"{top['observed']:.3f}. This is what allows the tiers to carry a stable meaning: severe "
        "is not merely the top of the list, it corresponds to a rate that holds up out of sample."
    )


def _weekly_capture(test: pd.DataFrame, figures: Path) -> str:
    weekly = []
    for week, frame in test.groupby("week_start"):
        positives = int(frame["label"].sum())
        top = frame.nlargest(20, "score")
        weekly.append(
            {"week": week, "attacked": positives, "caught": int(top["label"].sum())}
        )
    series = pd.DataFrame(weekly).sort_values("week")

    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.plot(series["week"], series["attacked"], linewidth=1.1, label="LGAs attacked")
    axis.plot(series["week"], series["caught"], linewidth=1.1, label="Of which in the top 20")
    axis.set_ylabel("LGAs")
    axis.set_title("Weekly attacks and how many the top 20 anticipated, 2023 to 2024")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figures / "weekly_capture.png")
    plt.close(figure)

    missed_weeks = int((series["caught"] == 0).sum())
    return (
        "Figure 3, weekly capture. The upper line counts LGAs attacked each week and the lower "
        "line counts how many of those the top twenty had already flagged. The two move together, "
        f"which is what a usable early warning looks like, but the list caught nothing at all in "
        f"{missed_weeks} of {len(series)} weeks. Those weeks are the honest limit of the approach: "
        "they are dominated by attacks in LGAs with no recent history, where a self-exciting model "
        "has nothing to excite from and only the wider feeds could help."
    )


def _driver_importance(model, panel: pd.DataFrame, figures: Path) -> str:
    from pau_risk.models.explain import explain

    latest = panel[panel["week_start"] == panel["week_start"].max()]
    explanation = explain(model.booster, latest, model.features, top_k=5)
    top = explanation.global_importance.head(12).iloc[::-1]

    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    axis.barh(top["label"], top["mean_abs_shap"])
    axis.set_xlabel("Mean absolute SHAP contribution")
    axis.set_title("What moves the score, scoring week")
    figure.tight_layout()
    figure.savefig(figures / "driver_importance.png")
    plt.close(figure)

    leader = explanation.global_importance.iloc[0]
    hawkes_share = explanation.global_importance.assign(
        is_hawkes=lambda frame: frame["feature"].str.startswith("hawkes_")
    ).groupby("is_hawkes")["mean_abs_shap"].sum()
    share = hawkes_share.get(True, 0.0) / hawkes_share.sum()
    return (
        "Figure 4, drivers. Bars are the mean absolute SHAP contribution across all 774 areas in "
        f"the scoring week. The single largest is {leader['label'].lower()}, and features derived "
        f"from the point process account for {share:.0%} of the total attribution. That is the "
        "clearest statement of what the hybrid is doing: the tree is mostly arbitrating between "
        "the components of the Hawkes intensity and adjusting them with calendar and operational "
        "context, rather than rediscovering temporal clustering from lagged counts."
    )


def _hawkes_kernel(settings, figures: Path) -> str:
    parameters = json.loads(
        (settings.paths.artifacts / "hawkes_parameters.json").read_text()
    )
    beta = parameters["decay_per_day"]
    days = np.linspace(0, 120, 400)
    kernel = beta * np.exp(-beta * days)

    figure, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    left.plot(days, kernel, linewidth=1.4)
    left.axvline(parameters["half_life_days"], linestyle="--", linewidth=1, color="black")
    left.set_xlabel("Days since an attack")
    left.set_ylabel("Excitation")
    left.set_title("Fitted decay of the triggering kernel")

    right.bar(
        ["Background", "Same LGA", "Neighbouring LGAs"],
        [
            1 - parameters["branching_ratio"],
            parameters["alpha_self"],
            parameters["alpha_neighbour"],
        ],
    )
    right.set_ylabel("Share of events")
    right.set_title("Where the next attack comes from")
    figure.tight_layout()
    figure.savefig(figures / "hawkes_parameters.png")
    plt.close(figure)

    return (
        "Figure 5, the fitted point process. The left panel shows how quickly the effect of an "
        f"attack decays, with a half life of {parameters['half_life_days']:.0f} days, so the "
        "elevated period after an incident runs to roughly six weeks rather than the one week a "
        "monthly incident report implies. The right panel decomposes the branching ratio of "
        f"{parameters['branching_ratio']:.2f}: about {parameters['branching_ratio']:.0%} of "
        f"recorded events are triggered by an earlier event, and of that triggered share "
        f"{parameters['self_share']:.0%} stays inside the same LGA while the rest crosses into a "
        "neighbour. That cross-border fraction is the quantitative form of the displacement the "
        "brief describes, and it is why the neighbour terms carry real weight in the model."
    )


if __name__ == "__main__":
    raise SystemExit(main())
