"""Rebuild the shinyapps.io bundle from the current warehouse and app.

The bundle is a derived artifact, not a second copy of the dashboard to maintain
by hand. Running this after a scoring run refreshes the data and re-derives the
app from app/app.R, so the deployed screen cannot drift away from the local one.
Three edits are applied during derivation, each forced by the platform: the data
path becomes flat, the reticulate-backed brief generation is swapped for the
stored brief, and the pinned host and port are removed because shinyapps.io
allocates the socket itself. The map needs no edit now that it renders through a
single sp based path instead of branching on whether sf happens to be installed.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "deploy" / "shinyapps"

WHOLE_TABLES = [
    "lga_registry", "model_runs", "predictions", "prediction_drivers",
    "threat_actor_edges", "security_briefs",
]
INDEXES = [
    "CREATE INDEX idx_pred_run ON predictions (run_id, probability DESC)",
    "CREATE INDEX idx_drv ON prediction_drivers (run_id, lga_code)",
    "CREATE INDEX idx_inc ON incidents (lga_code, event_date DESC)",
    "CREATE INDEX idx_actor ON threat_actor_edges (run_id, target_kind, target)",
]


def build_warehouse() -> tuple[int, float, float]:
    source_path = ROOT / "data" / "warehouse.db"
    target_path = BUNDLE / "data" / "warehouse.db"
    if not source_path.exists():
        raise SystemExit(f"no warehouse at {source_path}, run the pipeline first")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.unlink(missing_ok=True)

    source = sqlite3.connect(source_path)
    target = sqlite3.connect(target_path)
    try:
        for table in WHOLE_TABLES:
            ddl = source.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if ddl is None:
                raise SystemExit(f"table {table} is missing from the warehouse")
            target.execute(ddl[0])
            rows = source.execute(f"SELECT * FROM {table}").fetchall()
            if rows:
                placeholders = ",".join("?" * len(rows[0]))
                target.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)

        target.execute(
            """CREATE TABLE incidents (
                   event_id TEXT PRIMARY KEY, lga_code TEXT, event_date TEXT,
                   event_class TEXT, event_type TEXT, actor_primary TEXT,
                   actor_secondary TEXT, fatalities INTEGER, headline TEXT)"""
        )
        incidents = source.execute(
            """SELECT event_id, lga_code, event_date, event_class, event_type,
                      actor_primary, actor_secondary, fatalities, headline
               FROM incidents
               WHERE event_class = 'banditry_kidnapping' AND lga_code IS NOT NULL"""
        ).fetchall()
        target.executemany("INSERT INTO incidents VALUES (?,?,?,?,?,?,?,?,?)", incidents)

        for statement in INDEXES:
            target.execute(statement)
        target.commit()
        target.execute("VACUUM")
    finally:
        target.close()
        source.close()

    return (
        len(incidents),
        source_path.stat().st_size / 1e6,
        target_path.stat().st_size / 1e6,
    )


def derive_app() -> int:
    source = (ROOT / "app" / "app.R").read_text(encoding="utf-8")

    replacements = [
        (
            'source(file.path("R", "data_access.R"), local = TRUE)',
            'source("data_access.R", local = TRUE)',
        ),
        (
            """# The project root is explicit in a container and inferred when run from a
# checkout, where the app is normally started from inside app/.
resolve_root <- function() {
  from_env <- Sys.getenv("PAU_RISK_ROOT", "")
  if (nzchar(from_env)) return(normalizePath(from_env, mustWork = FALSE))
  parent <- normalizePath(file.path(getwd(), ".."), mustWork = FALSE)
  if (file.exists(file.path(parent, "data", "warehouse.db"))) return(parent)
  getwd()
}

CONFIG <- app_config(root = resolve_root())""",
            """# shinyapps.io deploys a single flat directory and sets the working directory to
# it, so the bundled data sits directly under the app root. No inference needed.
CONFIG <- app_config(root = getwd())""",
        ),
        (
            'actionButton("make_brief", "Draft brief for the current selection")',
            'actionButton("make_brief", "Show the stored security council brief")',
        ),
        (
            """# Cloud Run and most container hosts hand the port in through the environment
# and require binding on all interfaces rather than loopback.
shinyApp(
  ui, server,
  options = list(
    host = Sys.getenv("SHINY_HOST", "0.0.0.0"),
    port = as.integer(Sys.getenv("PORT", "7788"))
  )
)""",
            """# shinyapps.io allocates the port and binds the socket itself, so the app must
# not pin either. The container build does the opposite, which is why these two
# entry points differ on this one line.
shinyApp(ui, server)""",
        ),
    ]

    for old, new in replacements:
        if old not in source:
            raise SystemExit(
                "app/app.R no longer contains an expected block, so the bundle cannot "
                f"be derived safely. Missing:\n\n{old[:120]}..."
            )
        source = source.replace(old, new)

    (BUNDLE / "app.R").write_text(source, encoding="utf-8")
    return len(source.splitlines())


def main() -> int:
    kept, before, after = build_warehouse()
    shutil.copy(
        ROOT / "data" / "reference" / "nga_admin2_simplified.geojson",
        BUNDLE / "data" / "nga_admin2_simplified.geojson",
    )
    lines = derive_app()
    total = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file()) / 1e6
    print(f"app.R derived: {lines} lines")
    print(f"incidents kept: {kept}")
    print(f"warehouse: {before:.1f} MB -> {after:.1f} MB")
    print(f"bundle total: {total:.1f} MB")
    print(f"\nPublish with:\n  Rscript deploy/shinyapps/deploy.R")
    return 0


if __name__ == "__main__":
    sys.exit(main())
