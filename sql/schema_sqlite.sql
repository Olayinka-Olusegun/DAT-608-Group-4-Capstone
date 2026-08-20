-- Embedded fallback warehouse. Column names and semantics match
-- sql/schema_postgres.sql exactly so that queries written against one back end
-- run unchanged on the other. Geometry columns are replaced by the WKT text and
-- centroid scalars, and vector columns by a JSON-encoded array.

CREATE TABLE IF NOT EXISTS lga_registry (
    lga_code            TEXT PRIMARY KEY,
    lga_name            TEXT NOT NULL,
    state_code          TEXT NOT NULL,
    state_name          TEXT NOT NULL,
    zone                TEXT NOT NULL,
    senatorial_district TEXT,
    area_sqkm           REAL,
    centre_lat          REAL NOT NULL,
    centre_lon          REAL NOT NULL,
    geom_wkt            TEXT
);
CREATE INDEX IF NOT EXISTS idx_lga_registry_state ON lga_registry (state_name);

CREATE TABLE IF NOT EXISTS lga_adjacency (
    lga_code       TEXT NOT NULL,
    neighbour_code TEXT NOT NULL,
    weight         REAL NOT NULL,
    border_km      REAL,
    centroid_km    REAL,
    PRIMARY KEY (lga_code, neighbour_code)
);

CREATE TABLE IF NOT EXISTS incidents (
    event_id        TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    source_event_id TEXT,
    event_date      TEXT NOT NULL,
    event_class     TEXT NOT NULL,
    event_type      TEXT,
    actor_primary   TEXT,
    actor_secondary TEXT,
    dyad            TEXT,
    lga_code        TEXT,
    state_name      TEXT,
    latitude        REAL,
    longitude       REAL,
    geolocation_precision INTEGER,
    date_precision  INTEGER,
    fatalities      INTEGER DEFAULT 0,
    civilian_deaths INTEGER DEFAULT 0,
    victims         INTEGER,
    ransom_ngn      REAL,
    headline        TEXT,
    description     TEXT,
    source_url      TEXT,
    ingested_at     TEXT NOT NULL,
    geom_wkt        TEXT
);
CREATE INDEX IF NOT EXISTS idx_incidents_date ON incidents (event_date);
CREATE INDEX IF NOT EXISTS idx_incidents_lga_date ON incidents (lga_code, event_date);
CREATE INDEX IF NOT EXISTS idx_incidents_class ON incidents (event_class);

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    doc_type     TEXT,
    title        TEXT,
    published_at TEXT,
    url          TEXT,
    body         TEXT,
    lga_codes    TEXT,
    ransom_ngn   REAL,
    ingested_at  TEXT NOT NULL,
    embedding    TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_published ON documents (published_at);

CREATE TABLE IF NOT EXISTS chatter (
    chatter_id   TEXT PRIMARY KEY,
    platform     TEXT NOT NULL,
    posted_at    TEXT NOT NULL,
    lga_code     TEXT,
    body         TEXT,
    sentiment    REAL,
    threat_score REAL,
    url          TEXT,
    ingested_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chatter_lga_time ON chatter (lga_code, posted_at);

CREATE TABLE IF NOT EXISTS lga_week_features (
    lga_code   TEXT NOT NULL,
    week_start TEXT NOT NULL,
    features   TEXT NOT NULL,
    label      INTEGER,
    PRIMARY KEY (lga_code, week_start)
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id        TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    horizon_days  INTEGER NOT NULL,
    train_end     TEXT,
    metrics       TEXT,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id        TEXT NOT NULL,
    lga_code      TEXT NOT NULL,
    week_start    TEXT NOT NULL,
    probability   REAL NOT NULL,
    risk_tier     TEXT NOT NULL,
    rank_national INTEGER,
    rank_state    INTEGER,
    PRIMARY KEY (run_id, lga_code, week_start)
);
CREATE INDEX IF NOT EXISTS idx_predictions_week ON predictions (week_start, probability DESC);

CREATE TABLE IF NOT EXISTS prediction_drivers (
    run_id        TEXT NOT NULL,
    lga_code      TEXT NOT NULL,
    week_start    TEXT NOT NULL,
    driver_rank   INTEGER NOT NULL,
    feature_name  TEXT NOT NULL,
    feature_label TEXT NOT NULL,
    feature_value REAL,
    shap_value    REAL NOT NULL,
    PRIMARY KEY (run_id, lga_code, week_start, driver_rank)
);

CREATE TABLE IF NOT EXISTS threat_actor_edges (
    run_id      TEXT NOT NULL,
    actor       TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target      TEXT NOT NULL,
    events      INTEGER NOT NULL,
    fatalities  INTEGER NOT NULL,
    first_seen  TEXT,
    last_seen   TEXT,
    weight      REAL NOT NULL,
    PRIMARY KEY (run_id, actor, target_kind, target)
);

CREATE TABLE IF NOT EXISTS security_briefs (
    brief_id     TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    week_start   TEXT NOT NULL,
    scope        TEXT NOT NULL,
    generator    TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    content      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id      TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    lga_code      TEXT NOT NULL,
    week_start    TEXT NOT NULL,
    risk_tier     TEXT NOT NULL,
    probability   REAL NOT NULL,
    channel       TEXT NOT NULL,
    status        TEXT NOT NULL,
    dispatched_at TEXT NOT NULL,
    payload       TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_lga_time ON alerts (lga_code, dispatched_at DESC);
