from functools import lru_cache

from .calendar_ng import calendar_features, festival_within, is_dry_season, school_in_session
from .geography import (
    GEOPOLITICAL_ZONES,
    BoundarySet,
    assign_points_to_lgas,
    build_adjacency,
    download_boundaries,
    haversine_km,
    load_boundaries,
    write_simplified_geojson,
)


@lru_cache(maxsize=1)
def boundaries() -> BoundarySet:
    """Process-wide cache of the admin2 polygons, which are read repeatedly."""
    return load_boundaries()


__all__ = [
    "GEOPOLITICAL_ZONES",
    "BoundarySet",
    "assign_points_to_lgas",
    "boundaries",
    "build_adjacency",
    "calendar_features",
    "download_boundaries",
    "festival_within",
    "haversine_km",
    "is_dry_season",
    "load_boundaries",
    "school_in_session",
    "write_simplified_geojson",
]
