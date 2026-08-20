-- Warehouse schema for the LGA violence risk service (PostgreSQL target).
--
-- A single engine serves three access patterns: relational joins over the
-- incident panel, spatial adjacency queries through PostGIS, and similarity
-- search over report embeddings through pgvector. Extensions are created
-- conditionally so the file also applies on a stock PostgreSQL instance,
-- in which case the spatial and vector columns are simply omitted by the
-- loader and the equivalent scalar columns are used instead.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- reference
CREATE TABLE IF NOT EXISTS lga_registry (
    lga_code            TEXT PRIMARY KEY,
    lga_name            TEXT NOT NULL,
    state_code          TEXT NOT NULL,
    state_name          TEXT NOT NULL,
    zone                TEXT NOT NULL,
    senatorial_district TEXT,
    area_sqkm           DOUBLE PRECISION,
    centre_lat          DOUBLE PRECISION NOT NULL,
    centre_lon          DOUBLE PRECISION NOT NULL,
    geom                GEOMETRY(MultiPolygon, 4326)
);
CREATE INDEX IF NOT EXISTS idx_lga_registry_state ON lga_registry (state_name);
CREATE INDEX IF NOT EXISTS idx_lga_registry_geom ON lga_registry USING GIST (geom);

CREATE TABLE IF NOT EXISTS lga_adjacency (
    lga_code       TEXT NOT NULL REFERENCES lga_registry (lga_code),
    neighbour_code TEXT NOT NULL REFERENCES lga_registry (lga_code),
    weight         DOUBLE PRECISION NOT NULL,
    border_km      DOUBLE PRECISION,
    centroid_km    DOUBLE PRECISION,
    PRIMARY KEY (lga_code, neighbour_code)
);

-- ------------------------------------------------------------------- events
CREATE TABLE IF NOT EXISTS incidents (
    event_id        TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    source_event_id TEXT,
    event_date      DATE NOT NULL,
    event_class     TEXT NOT NULL,          -- banditry_kidnapping | state_operation | other
    event_type      TEXT,
    actor_primary   TEXT,
    actor_secondary TEXT,
    dyad            TEXT,
    lga_code        TEXT REFERENCES lga_registry (lga_code),
    state_name      TEXT,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    geolocation_precision SMALLINT,
    date_precision  SMALLINT,
    fatalities      INTEGER DEFAULT 0,
    civilian_deaths INTEGER DEFAULT 0,
    victims         INTEGER,
    ransom_ngn      NUMERIC,
    headline        TEXT,
    description     TEXT,
    source_url      TEXT,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    geom            GEOMETRY(Point, 4326)
);
CREATE INDEX IF NOT EXISTS idx_incidents_date ON incidents (event_date);
CREATE INDEX IF NOT EXISTS idx_incidents_lga_date ON incidents (lga_code, event_date);
CREATE INDEX IF NOT EXISTS idx_incidents_class ON incidents (event_class);
CREATE INDEX IF NOT EXISTS idx_incidents_geom ON incidents USING GIST (geom);

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    doc_type     TEXT,
    title        TEXT,
    published_at DATE,
    url          TEXT,
    body         TEXT,
    lga_codes    TEXT[],
    ransom_ngn   NUMERIC,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding    VECTOR(384)
);
CREATE INDEX IF NOT EXISTS idx_documents_published ON documents (published_at);

CREATE TABLE IF NOT EXISTS chatter (
    chatter_id  TEXT PRIMARY KEY,
    platform    TEXT NOT NULL,
    posted_at   TIMESTAMPTZ NOT NULL,
    lga_code    TEXT REFERENCES lga_registry (lga_code),
    body        TEXT,
    sentiment   DOUBLE PRECISION,
    threat_score DOUBLE PRECISION,
    url         TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chatter_lga_time ON chatter (lga_code, posted_at);

-- ------------------------------------------------------------ feature store
CREATE TABLE IF NOT EXISTS lga_week_features (
    lga_code   TEXT NOT NULL REFERENCES lga_registry (lga_code),
    week_start DATE NOT NULL,
    features   JSONB NOT NULL,
    label      SMALLINT,
    PRIMARY KEY (lga_code, week_start)
);

-- --------------------------------------------------------------- prediction
CREATE TABLE IF NOT EXISTS model_runs (
    run_id        TEXT PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version TEXT NOT NULL,
    horizon_days  INTEGER NOT NULL,
    train_end     DATE,
    metrics       JSONB,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id       TEXT NOT NULL REFERENCES model_runs (run_id),
    lga_code     TEXT NOT NULL REFERENCES lga_registry (lga_code),
    week_start   DATE NOT NULL,
    probability  DOUBLE PRECISION NOT NULL,
    risk_tier    TEXT NOT NULL,
    rank_national INTEGER,
    rank_state    INTEGER,
    PRIMARY KEY (run_id, lga_code, week_start)
);
CREATE INDEX IF NOT EXISTS idx_predictions_week ON predictions (week_start, probability DESC);

CREATE TABLE IF NOT EXISTS prediction_drivers (
    run_id        TEXT NOT NULL,
    lga_code      TEXT NOT NULL,
    week_start    DATE NOT NULL,
    driver_rank   SMALLINT NOT NULL,
    feature_name  TEXT NOT NULL,
    feature_label TEXT NOT NULL,
    feature_value DOUBLE PRECISION,
    shap_value    DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, lga_code, week_start, driver_rank)
);

CREATE TABLE IF NOT EXISTS threat_actor_edges (
    run_id       TEXT NOT NULL,
    actor        TEXT NOT NULL,
    target_kind  TEXT NOT NULL,   -- lga | state | community
    target       TEXT NOT NULL,
    events       INTEGER NOT NULL,
    fatalities   INTEGER NOT NULL,
    first_seen   DATE,
    last_seen    DATE,
    weight       DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, actor, target_kind, target)
);

CREATE TABLE IF NOT EXISTS security_briefs (
    brief_id     TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    week_start   DATE NOT NULL,
    scope        TEXT NOT NULL,   -- national or a state name
    generator    TEXT NOT NULL,   -- claude model id or 'deterministic-template'
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    content      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id    TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    lga_code    TEXT NOT NULL,
    week_start  DATE NOT NULL,
    risk_tier   TEXT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    channel     TEXT NOT NULL,
    status      TEXT NOT NULL,
    dispatched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload     JSONB
);
CREATE INDEX IF NOT EXISTS idx_alerts_lga_time ON alerts (lga_code, dispatched_at DESC);
