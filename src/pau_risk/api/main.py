"""The service layer.

Two kinds of consumer meet here. Upstream, connectors that cannot be pulled on a
schedule push into the ingestion endpoints, which validate against the same Avro
contracts the batch producers use and publish onto the same topics; a field
report submitted this way is indistinguishable downstream from one that was
scraped. Downstream, the Shiny dashboard, the alerting job and any other client
read predictions, drivers, the actor graph and the brief.

Read endpoints serve whatever the most recent model run wrote, and always say
which run that was, so a screen and a brief opened minutes apart cannot silently
disagree about which model they are showing.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ..config import get_settings
from ..logging_utils import configure, get_logger
from ..storage import (
    latest_run_id,
    load_drivers,
    load_predictions,
    load_registry,
    read_sql,
    table_counts,
)
from ..stream import EventBus

configure()
LOGGER = get_logger(__name__)

app = FastAPI(
    title="LGA Violence Risk Service",
    version="0.1.0",
    description=(
        "Seven day banditry and kidnapping risk for the 774 Nigerian local "
        "government areas, with the drivers behind each score."
    ),
)


# ------------------------------------------------------------------- models
class DocumentIn(BaseModel):
    source: str = Field(..., examples=["field_report"])
    title: str
    body: str
    url: str | None = None
    published_at: date | None = None
    doc_type: str | None = "field_report"


class ChatterIn(BaseModel):
    platform: str = Field(..., examples=["telegram"])
    body: str
    posted_at: datetime
    lga_code: str | None = None
    url: str | None = None


# -------------------------------------------------------------- diagnostics
@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    counts = table_counts()
    return {
        "status": "ok",
        "run_id": latest_run_id(),
        "tables": counts,
        "kafka_bootstrap": settings.kafka_bootstrap,
    }


@app.get("/meta/run")
def meta_run() -> dict[str, Any]:
    frame = read_sql(
        "SELECT run_id, created_at, model_version, horizon_days, train_end, metrics "
        "FROM model_runs ORDER BY created_at DESC LIMIT 1"
    )
    if frame.empty:
        raise HTTPException(status_code=404, detail="no model run has been recorded")
    row = frame.iloc[0].to_dict()
    if isinstance(row.get("metrics"), str):
        row["metrics"] = json.loads(row["metrics"])
    return row


# ------------------------------------------------------------------ reading
@app.get("/predictions/latest")
def predictions_latest(
    state: str | None = Query(default=None, description="Filter to one state"),
    tier: str | None = Query(default=None, description="Low, Elevated, High or Severe"),
    limit: int = Query(default=100, ge=1, le=800),
) -> dict[str, Any]:
    frame = load_predictions()
    if frame.empty:
        raise HTTPException(status_code=404, detail="no predictions are available")
    if state:
        frame = frame[frame["state_name"].str.lower() == state.lower()]
    if tier:
        frame = frame[frame["risk_tier"].str.lower() == tier.lower()]
    return {
        "run_id": frame["run_id"].iloc[0] if not frame.empty else latest_run_id(),
        "count": int(len(frame)),
        "results": json.loads(frame.head(limit).to_json(orient="records", date_format="iso")),
    }


@app.get("/predictions/{lga_code}")
def prediction_for_lga(lga_code: str) -> dict[str, Any]:
    frame = load_predictions()
    row = frame[frame["lga_code"] == lga_code]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"no prediction for {lga_code}")
    record = json.loads(row.iloc[[0]].to_json(orient="records", date_format="iso"))[0]
    drivers = load_drivers(record["run_id"], lga_code)
    record["drivers"] = json.loads(drivers.to_json(orient="records", date_format="iso"))
    return record


@app.get("/predictions/{lga_code}/drivers")
def drivers_for_lga(lga_code: str) -> dict[str, Any]:
    run_id = latest_run_id()
    if run_id is None:
        raise HTTPException(status_code=404, detail="no model run has been recorded")
    frame = load_drivers(run_id, lga_code)
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"no drivers stored for {lga_code}")
    return {
        "run_id": run_id,
        "lga_code": lga_code,
        "drivers": json.loads(frame.to_json(orient="records", date_format="iso")),
    }


@app.get("/graph/actors")
def actor_graph(
    target_kind: str = Query(default="lga", pattern="^(lga|state)$"),
    target: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    run_id = latest_run_id()
    query = (
        "SELECT run_id, actor, target_kind, target, events, fatalities, "
        "first_seen, last_seen, weight FROM threat_actor_edges "
        "WHERE run_id = :run_id AND target_kind = :kind"
    )
    params: dict[str, Any] = {"run_id": run_id, "kind": target_kind}
    if target:
        query += " AND target = :target"
        params["target"] = target
    query += " ORDER BY weight DESC"
    frame = read_sql(query, params)
    return {
        "run_id": run_id,
        "count": int(len(frame)),
        "edges": json.loads(frame.head(limit).to_json(orient="records")),
    }


@app.get("/brief/latest")
def brief_latest(scope: str = "national") -> dict[str, Any]:
    frame = read_sql(
        "SELECT brief_id, run_id, week_start, scope, generator, generated_at, content "
        "FROM security_briefs WHERE scope = :scope ORDER BY generated_at DESC LIMIT 1",
        {"scope": scope},
    )
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"no brief stored for scope {scope}")
    return json.loads(frame.to_json(orient="records", date_format="iso"))[0]


@app.post("/brief/generate")
def brief_generate(scope: str = Body(default="national", embed=True)) -> dict[str, Any]:
    from .. import brief as brief_module

    run_id = latest_run_id()
    if run_id is None:
        raise HTTPException(status_code=404, detail="no model run has been recorded")
    predictions = load_predictions(run_id)
    drivers = load_drivers(run_id)
    edges = read_sql(
        "SELECT actor, target_kind, target, events, fatalities, first_seen, last_seen, weight "
        "FROM threat_actor_edges WHERE run_id = :run_id",
        {"run_id": run_id},
    )
    week_start = pd.to_datetime(predictions["week_start"].iloc[0]).date()
    generated = brief_module.create_and_store(
        run_id=run_id,
        week_start=week_start,
        predictions=predictions,
        drivers=drivers,
        actor_edges=edges,
        scope=scope,
    )
    return {
        "brief_id": generated.brief_id,
        "generator": generated.generator,
        "scope": generated.scope,
        "content": generated.content,
    }


@app.get("/geo/lgas")
def geojson(state: str | None = None) -> dict[str, Any]:
    """Boundaries joined to the current scores, ready for the leaflet layer."""
    settings = get_settings()
    path = settings.paths.reference / settings.section("reference")["boundaries_member"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="boundary file has not been built")
    payload = json.loads(path.read_text(encoding="utf-8"))

    scores = load_predictions()
    lookup = (
        scores.set_index("lga_code")[["probability", "risk_tier", "rank_national"]].to_dict("index")
        if not scores.empty
        else {}
    )
    features = []
    for feature in payload["features"]:
        properties = feature["properties"]
        code = properties.get("adm2_pcode")
        if state and str(properties.get("adm1_name", "")).lower() != state.lower():
            continue
        scored = lookup.get(code, {})
        feature["properties"] = {
            "lga_code": code,
            "lga_name": properties.get("adm2_name"),
            "state_name": properties.get("adm1_name"),
            "probability": scored.get("probability"),
            "risk_tier": scored.get("risk_tier", "Unscored"),
            "rank_national": scored.get("rank_national"),
        }
        features.append(feature)
    return {"type": "FeatureCollection", "features": features}


@app.get("/reference/registry")
def registry() -> dict[str, Any]:
    frame = load_registry()
    return {"count": int(len(frame)), "results": json.loads(frame.to_json(orient="records"))}


# ---------------------------------------------------------------- ingestion
@app.post("/ingest/document", status_code=202)
def ingest_document(document: DocumentIn) -> dict[str, Any]:
    """Accept a report that no scheduled connector covers."""
    from ..ingest.base import stable_id, utc_now
    from ..nlp.extract import Gazetteer, extract
    from ..reference import boundaries

    facts = extract(f"{document.title}. {document.body}", Gazetteer.from_registry(boundaries().frame))
    record = {
        "doc_id": stable_id("api", document.url, document.title),
        "source": document.source,
        "doc_type": document.doc_type,
        "title": document.title,
        "published_at": (document.published_at or date.today()),
        "url": document.url,
        "body": document.body[:20000],
        "lga_codes": facts.lga_codes,
        "ransom_ngn": facts.ransom_ngn,
        "ingested_at": utc_now(),
    }
    bus = EventBus()
    published = bus.publish("document", [record], producer_name="api")
    bus.close()
    return {
        "accepted": published.count,
        "transport": published.transport,
        "matched_lgas": facts.lga_codes,
        "ransom_ngn": facts.ransom_ngn,
    }


@app.post("/ingest/chatter", status_code=202)
def ingest_chatter(post: ChatterIn) -> dict[str, Any]:
    from ..ingest.base import stable_id, utc_now
    from ..nlp.extract import Gazetteer, extract
    from ..reference import boundaries

    facts = extract(post.body, Gazetteer.from_registry(boundaries().frame))
    record = {
        "chatter_id": stable_id("api", post.url, post.posted_at.isoformat()),
        "platform": post.platform,
        "posted_at": post.posted_at.isoformat(),
        "lga_code": post.lga_code or (facts.lga_codes[0] if facts.lga_codes else None),
        "body": post.body[:2000],
        "sentiment": facts.sentiment,
        "threat_score": facts.threat_score,
        "url": post.url,
        "ingested_at": utc_now(),
    }
    bus = EventBus()
    published = bus.publish("chatter", [record], producer_name="api")
    bus.close()
    return {
        "accepted": published.count,
        "transport": published.transport,
        "threat_score": facts.threat_score,
        "matched_lga": record["lga_code"],
    }


@app.post("/alerts/run")
def alerts_run(send: bool = Body(default=False, embed=True)) -> dict[str, Any]:
    from ..alerting import run as run_alerts

    run_id = latest_run_id()
    if run_id is None:
        raise HTTPException(status_code=404, detail="no model run has been recorded")
    predictions = load_predictions(run_id)
    week_start = pd.to_datetime(predictions["week_start"].iloc[0]).date()
    decisions = run_alerts(run_id, week_start, predictions, send=send)
    return {
        "run_id": run_id,
        "week_start": week_start.isoformat(),
        "decisions": json.loads(decisions.to_json(orient="records")),
    }
