"""Command line entry point for every stage of the pipeline.

Each stage can be run on its own, which is how the components are developed and
debugged, and ``pipeline`` runs them in order, which is what the weekly schedule
calls. Stages communicate only through the warehouse and the artifact directory,
so a stage can be re-run without re-running the ones before it.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

import pandas as pd

from .config import get_settings
from .logging_utils import configure, get_logger

LOGGER = get_logger("pau_risk.cli")


# ------------------------------------------------------------------- stages
def cmd_reference(args: argparse.Namespace) -> int:
    from .reference import build_adjacency, load_boundaries, write_simplified_geojson
    from .storage import get_engine, init_schema, upsert_frame

    settings = get_settings()
    init_schema(settings)
    boundaries = load_boundaries(settings)
    registry = boundaries.frame.copy()
    adjacency = build_adjacency(boundaries, settings)

    _, backend = get_engine(settings)
    if backend.is_postgres and backend.postgis:
        registry["geom"] = [boundaries.geometries[code].wkt for code in registry["lga_code"]]
    else:
        registry["geom_wkt"] = None

    upsert_frame("lga_registry", registry)
    upsert_frame("lga_adjacency", adjacency)
    write_simplified_geojson(boundaries, settings)
    LOGGER.info("reference loaded: %d LGAs, %d edges", len(registry), len(adjacency))
    return 0


def _default_since(until: date, days: int) -> date:
    """Backfill on a cold warehouse, otherwise pull only the recent window.

    A weekly refresh only needs the last fortnight, and pulling the whole history
    every Monday would be wasteful. On an empty warehouse that same short window
    produces a panel with no events in it, which is how a run that trains on
    nothing and marks every area severe becomes possible. So the window is chosen
    by looking at what is already stored rather than by a fixed default.
    """
    from .config import get_settings
    from .storage import init_schema, read_sql

    settings = get_settings()
    init_schema(settings)
    try:
        stored = int(read_sql("SELECT COUNT(*) AS n FROM incidents").iloc[0]["n"])
    except Exception:  # noqa: BLE001 - a missing table means a cold start
        stored = 0
    if stored == 0:
        panel_start = date.fromisoformat(str(settings.section("features")["panel_start"]))
        backfill_from = panel_start - timedelta(days=365 * 3)
        LOGGER.info(
            "no incidents stored, backfilling from %s to cover the panel period",
            backfill_from,
        )
        return backfill_from
    return until - timedelta(days=days)


def cmd_ingest(args: argparse.Namespace) -> int:
    from .ingest import drain_to_warehouse, run_ingestion

    until = date.fromisoformat(args.until) if args.until else date.today()
    since = (
        date.fromisoformat(args.since)
        if args.since
        else _default_since(until, args.days)
    )
    summary = run_ingestion(since=since, until=until, only=args.only)
    if not summary.empty:
        print(summary.to_string(index=False))
    if not args.no_drain:
        print(json.dumps(drain_to_warehouse(), indent=2))
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    from .features import build_panel
    from .storage import load_adjacency, load_incidents, load_registry, read_sql

    settings = get_settings()
    documents = read_sql("SELECT published_at, lga_codes, ransom_ngn FROM documents")
    chatter = read_sql("SELECT posted_at, lga_code, threat_score FROM chatter")
    result = build_panel(
        load_incidents(), load_registry(), load_adjacency(), documents, chatter, settings
    )
    path = settings.paths.processed / "panel.parquet"
    result.frame.to_parquet(path)
    result.coverage.to_csv(settings.paths.artifacts / "feature_coverage.csv", index=False)
    (settings.paths.artifacts / "hawkes_parameters.json").write_text(
        json.dumps(result.hawkes_parameters, indent=2)
    )
    LOGGER.info("panel written to %s", path)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .models import train

    settings = get_settings()
    panel = pd.read_parquet(settings.paths.processed / "panel.parquet")
    coverage = pd.read_csv(settings.paths.artifacts / "feature_coverage.csv")
    hawkes = json.loads((settings.paths.artifacts / "hawkes_parameters.json").read_text())
    features = coverage.loc[coverage["in_model"], "feature"].tolist()
    model = train(panel, features, hawkes, coverage, settings)
    print(model.comparison.round(4).to_string(index=False))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from . import graph as graph_module
    from .models import load, persist, score_week
    from .storage import load_incidents, load_registry

    settings = get_settings()
    panel = pd.read_parquet(settings.paths.processed / "panel.parquet")
    model = load(settings)
    week = pd.Timestamp(args.week) if args.week else None
    run = score_week(model, panel, week, settings)
    persist(run, model, notes=args.notes or "")

    threat = graph_module.build(load_incidents(), load_registry(), run.run_id, run.week_start)
    graph_module.persist(threat)

    columns = ["rank_national", "lga_name", "state_name", "probability", "risk_tier"]
    print(run.predictions[columns].head(args.top).to_string(index=False))
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    from . import brief as brief_module
    from .storage import latest_run_id, load_drivers, load_predictions, read_sql

    run_id = latest_run_id()
    if run_id is None:
        LOGGER.error("no model run recorded, run score first")
        return 1
    predictions = load_predictions(run_id)
    drivers = load_drivers(run_id)
    edges = read_sql(
        "SELECT actor, target_kind, target, events, fatalities, first_seen, last_seen, weight "
        "FROM threat_actor_edges WHERE run_id = :run_id",
        {"run_id": run_id},
    )
    week_start = pd.to_datetime(predictions["week_start"].iloc[0]).date()
    generated = brief_module.create_and_store(
        run_id, week_start, predictions, drivers, edges, scope=args.scope
    )
    settings = get_settings()
    output = settings.paths.artifacts / f"brief-{args.scope.replace(' ', '_')}.txt"
    output.write_text(generated.content, encoding="utf-8")
    print(generated.content)
    LOGGER.info("brief written to %s (generator: %s)", output, generated.generator)
    return 0


def cmd_alert(args: argparse.Namespace) -> int:
    from .alerting import run as run_alerts
    from .storage import latest_run_id, load_predictions

    run_id = latest_run_id()
    if run_id is None:
        LOGGER.error("no model run recorded, run score first")
        return 1
    predictions = load_predictions(run_id)
    week_start = pd.to_datetime(predictions["week_start"].iloc[0]).date()
    decisions = run_alerts(run_id, week_start, predictions, send=args.send)
    if decisions.empty:
        print("No LGA cleared the alerting threshold this week.")
    else:
        print(decisions.to_string(index=False))
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    stages = (
        ("reference", cmd_reference),
        ("ingest", cmd_ingest),
        ("features", cmd_features),
        ("train", cmd_train),
        ("score", cmd_score),
        ("brief", cmd_brief),
        ("alert", cmd_alert),
    )
    for name, handler in stages:
        if args.skip and name in args.skip:
            LOGGER.info("skipping stage %s", name)
            continue
        LOGGER.info("stage %s", name)
        code = handler(args)
        if code != 0:
            LOGGER.error("stage %s failed", name)
            return code
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .storage import latest_run_id, table_counts

    settings = get_settings()
    print(json.dumps({
        "run_id": latest_run_id(),
        "tables": table_counts(),
        "mlflow": settings.mlflow_uri,
        "kafka": settings.kafka_bootstrap,
    }, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("pau_risk.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


# --------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pau-risk", description=__doc__)
    parser.add_argument("--log-level", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("reference", help="build the LGA registry and adjacency graph")

    ingest = subparsers.add_parser("ingest", help="run the producers and load the warehouse")
    ingest.add_argument("--since")
    ingest.add_argument("--until")
    ingest.add_argument("--days", type=int, default=30)
    ingest.add_argument("--only", nargs="*")
    ingest.add_argument("--no-drain", action="store_true")

    subparsers.add_parser("features", help="build the LGA by week feature panel")
    subparsers.add_parser("train", help="train, calibrate and evaluate the hybrid model")

    score = subparsers.add_parser("score", help="score the forecast week")
    score.add_argument("--week")
    score.add_argument("--top", type=int, default=15)
    score.add_argument("--notes", default="")

    brief = subparsers.add_parser("brief", help="draft the security council brief")
    brief.add_argument("--scope", default="national")

    alert = subparsers.add_parser("alert", help="evaluate and dispatch alerts")
    alert.add_argument("--send", action="store_true", help="actually dispatch, not a dry run")

    pipeline = subparsers.add_parser("pipeline", help="run every stage in order")
    pipeline.add_argument("--since")
    pipeline.add_argument("--until")
    pipeline.add_argument("--days", type=int, default=30)
    pipeline.add_argument("--only", nargs="*")
    pipeline.add_argument("--no-drain", action="store_true")
    pipeline.add_argument("--week")
    pipeline.add_argument("--top", type=int, default=15)
    pipeline.add_argument("--notes", default="scheduled pipeline run")
    pipeline.add_argument("--scope", default="national")
    pipeline.add_argument("--send", action="store_true")
    pipeline.add_argument("--skip", nargs="*", default=[])

    subparsers.add_parser("status", help="report warehouse and run state")

    serve = subparsers.add_parser("serve", help="run the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    return parser


HANDLERS = {
    "reference": cmd_reference,
    "ingest": cmd_ingest,
    "features": cmd_features,
    "train": cmd_train,
    "score": cmd_score,
    "brief": cmd_brief,
    "alert": cmd_alert,
    "pipeline": cmd_pipeline,
    "status": cmd_status,
    "serve": cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure(args.log_level)
    return HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
