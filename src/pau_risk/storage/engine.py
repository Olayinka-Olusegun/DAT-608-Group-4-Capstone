"""Warehouse connection handling.

The architecture in the brief specifies PostgreSQL with PostGIS and pgvector.
That remains the primary target: set ``DATABASE_URL`` and the pipeline uses it,
including the spatial and vector columns. When no reachable PostgreSQL instance
is configured the same logical schema is created in an embedded SQLite file,
which is the CSV or JSON style back end the brief allows as an alternative. Every
downstream query is written against the shared column names, so switching back
ends does not change any calling code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, create_engine, inspect, text

from ..config import Settings, get_settings
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class Backend:
    """Describes the capabilities actually available on the live connection."""

    name: str           # postgresql or sqlite
    url: str
    postgis: bool
    pgvector: bool

    @property
    def is_postgres(self) -> bool:
        return self.name == "postgresql"

    @property
    def geometry_column(self) -> str:
        return "geom" if self.postgis else "geom_wkt"


_ENGINE: Engine | None = None
_BACKEND: Backend | None = None


def _probe_postgres(url: str) -> Engine | None:
    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return engine
    except Exception as exc:  # noqa: BLE001 - any driver or network failure means fall back
        LOGGER.warning("PostgreSQL at the configured URL is not usable (%s)", exc.__class__.__name__)
        return None


def _extension_available(engine: Engine, extension: str) -> bool:
    try:
        with engine.begin() as connection:
            connection.execute(text(f"CREATE EXTENSION IF NOT EXISTS {extension}"))
        return True
    except Exception:  # noqa: BLE001
        return False


def get_engine(settings: Settings | None = None) -> tuple[Engine, Backend]:
    """Return the shared engine and a description of what the back end supports."""
    global _ENGINE, _BACKEND
    if _ENGINE is not None and _BACKEND is not None:
        return _ENGINE, _BACKEND

    settings = settings or get_settings()
    url = settings.database_url
    engine = _probe_postgres(url) if url else None

    if engine is not None:
        backend = Backend(
            name="postgresql",
            url=url or "",
            postgis=_extension_available(engine, "postgis"),
            pgvector=_extension_available(engine, "vector"),
        )
        LOGGER.info(
            "warehouse=postgresql postgis=%s pgvector=%s", backend.postgis, backend.pgvector
        )
    else:
        sqlite_path = settings.sqlite_path
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_url = f"sqlite:///{sqlite_path}"
        engine = create_engine(sqlite_url)
        backend = Backend(name="sqlite", url=sqlite_url, postgis=False, pgvector=False)
        LOGGER.info("warehouse=sqlite path=%s", sqlite_path)

    _ENGINE, _BACKEND = engine, backend
    return engine, backend


def reset_engine() -> None:
    """Drop the cached engine. Used by the tests to switch back ends."""
    global _ENGINE, _BACKEND
    if _ENGINE is not None:
        _ENGINE.dispose()
    _ENGINE, _BACKEND = None, None


def _schema_file(backend: Backend, settings: Settings) -> Path:
    name = "schema_postgres.sql" if backend.is_postgres else "schema_sqlite.sql"
    return settings.paths.root / "sql" / name


def _split_statements(script: str) -> list[str]:
    statements: list[str] = []
    for chunk in script.split(";"):
        cleaned = "\n".join(
            line for line in chunk.splitlines() if not line.strip().startswith("--")
        ).strip()
        if cleaned:
            statements.append(cleaned)
    return statements


def init_schema(settings: Settings | None = None) -> Backend:
    """Create every table, tolerating the absence of optional extensions."""
    settings = settings or get_settings()
    engine, backend = get_engine(settings)
    script = _schema_file(backend, settings).read_text(encoding="utf-8")

    for statement in _split_statements(script):
        lowered = statement.lower()
        if backend.is_postgres:
            if "create extension" in lowered:
                continue  # already probed
            if not backend.postgis and ("geometry(" in lowered or "using gist" in lowered):
                statement = _strip_unsupported(statement, "geom")
                if statement is None:
                    continue
            if not backend.pgvector and "vector(" in lowered:
                statement = _strip_unsupported(statement, "embedding")
                if statement is None:
                    continue
        try:
            with engine.begin() as connection:
                connection.execute(text(statement))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("DDL failed: %s | %s", statement.splitlines()[0][:80], exc)
            raise
    return backend


def _strip_unsupported(statement: str, column: str) -> str | None:
    """Remove a column or index that depends on an unavailable extension."""
    if statement.lower().startswith("create index"):
        return None if column in statement.lower() else statement
    kept = [
        line
        for line in statement.splitlines()
        if not line.strip().lower().startswith(column)
    ]
    joined = "\n".join(kept)
    return joined.replace(",\n)", "\n)")


def read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    engine, _ = get_engine()
    with engine.connect() as connection:
        return pd.read_sql(text(query), connection, params=params or {})


def execute(statement: str, params: dict | list[dict] | None = None) -> None:
    engine, _ = get_engine()
    with engine.begin() as connection:
        connection.execute(text(statement), params or {})
