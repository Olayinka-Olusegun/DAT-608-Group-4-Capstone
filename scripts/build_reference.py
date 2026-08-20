"""Populate the LGA registry and adjacency graph in the warehouse.

Run this once before anything else. Everything downstream joins on ``lga_code``,
so the registry is the fixed point of the whole system.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pau_risk.config import get_settings  # noqa: E402
from pau_risk.logging_utils import configure, get_logger  # noqa: E402
from pau_risk.reference import (  # noqa: E402
    build_adjacency,
    load_boundaries,
    write_simplified_geojson,
)
from pau_risk.storage import get_engine, init_schema, upsert_frame  # noqa: E402

LOGGER = get_logger("build_reference")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true")
    arguments = parser.parse_args()

    configure()
    settings = get_settings()
    backend = init_schema(settings)

    boundaries = load_boundaries(settings)
    registry = boundaries.frame.copy()
    adjacency = build_adjacency(boundaries, settings)

    _, backend_info = get_engine(settings)
    if backend_info.is_postgres and backend_info.postgis:
        registry["geom"] = [
            boundaries.geometries[code].wkt for code in registry["lga_code"]
        ]
    else:
        registry["geom_wkt"] = None  # polygons are served from the GeoJSON, not the row

    written_registry = upsert_frame("lga_registry", registry)
    written_adjacency = upsert_frame("lga_adjacency", adjacency)
    write_simplified_geojson(boundaries, settings)

    degree = adjacency.groupby("lga_code").size()
    LOGGER.info(
        "registry rows=%d adjacency rows=%d mean_degree=%.2f min_degree=%d max_degree=%d",
        written_registry,
        written_adjacency,
        degree.mean(),
        degree.min(),
        degree.max(),
    )
    LOGGER.info(
        "states=%d zones=%d backend=%s",
        registry["state_name"].nunique(),
        registry["zone"].nunique(),
        backend.name,
    )

    export = settings.paths.reference / "lga_registry.parquet"
    registry.drop(columns=[c for c in ("geom", "geom_wkt") if c in registry], errors="ignore").to_parquet(export)
    adjacency.to_parquet(settings.paths.reference / "lga_adjacency.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
