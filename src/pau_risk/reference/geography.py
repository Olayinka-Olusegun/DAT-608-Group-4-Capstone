"""Build the LGA registry and the spatial adjacency graph.

Geography is taken from the OCHA Common Operational Dataset for Nigerian
administrative boundaries, which carries all 774 admin level 2 units with their
official P-codes, parent state, senatorial district, area and centroid. The same
P-code scheme is used by GRID3, so the adjacency graph produced here is the
GRID3 adjacency the modelling section of the brief calls for.

Adjacency is derived from the polygons themselves rather than from centroid
distance, because contagion in banditry follows shared borders and the road
network across them, not straight-line proximity. Each edge is weighted by the
share of an LGA's total border that it accounts for, so the weights entering the
Hawkes kernel already encode how exposed a unit is to each neighbour.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from ..config import Settings, get_settings
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

EARTH_KM_PER_DEGREE = 111.32

GEOPOLITICAL_ZONES: dict[str, str] = {
    "Benue": "North Central", "Federal Capital Territory": "North Central",
    "Kogi": "North Central", "Kwara": "North Central", "Nasarawa": "North Central",
    "Niger": "North Central", "Plateau": "North Central",
    "Adamawa": "North East", "Bauchi": "North East", "Borno": "North East",
    "Gombe": "North East", "Taraba": "North East", "Yobe": "North East",
    "Jigawa": "North West", "Kaduna": "North West", "Kano": "North West",
    "Katsina": "North West", "Kebbi": "North West", "Sokoto": "North West",
    "Zamfara": "North West",
    "Abia": "South East", "Anambra": "South East", "Ebonyi": "South East",
    "Enugu": "South East", "Imo": "South East",
    "Akwa Ibom": "South South", "Bayelsa": "South South", "Cross River": "South South",
    "Delta": "South South", "Edo": "South South", "Rivers": "South South",
    "Ekiti": "South West", "Lagos": "South West", "Ogun": "South West",
    "Ondo": "South West", "Osun": "South West", "Oyo": "South West",
}

# The COD file spells the capital territory differently from common usage.
STATE_ALIASES = {
    "Abuja Federal Capital Territory": "Federal Capital Territory",
    "FCT": "Federal Capital Territory",
    "Nasarawa": "Nasarawa",
    "Akwa Ibom": "Akwa Ibom",
}


@dataclass
class BoundarySet:
    frame: pd.DataFrame
    geometries: dict[str, object]


def download_boundaries(settings: Settings | None = None, force: bool = False) -> Path:
    """Fetch the COD administrative boundary archive and return the admin2 file."""
    settings = settings or get_settings()
    reference_cfg = settings.section("reference")
    archive = settings.paths.raw / "nga_admin_boundaries.geojson.zip"
    member = reference_cfg["boundaries_member"]
    target = settings.paths.reference / member

    if target.exists() and not force:
        return target

    if not archive.exists() or force:
        url = reference_cfg["boundaries_url"]
        LOGGER.info("downloading administrative boundaries from %s", url)
        response = requests.get(url, timeout=600, stream=True)
        response.raise_for_status()
        archive.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("wb") as handle:
            for block in response.iter_content(chunk_size=1 << 20):
                handle.write(block)

    with zipfile.ZipFile(archive) as bundle:
        target.parent.mkdir(parents=True, exist_ok=True)
        with bundle.open(member) as source, target.open("wb") as sink:
            sink.write(source.read())
    LOGGER.info("admin2 boundaries written to %s", target)
    return target


def _normalise_state(name: str) -> str:
    cleaned = (name or "").strip()
    return STATE_ALIASES.get(cleaned, cleaned)


def load_boundaries(settings: Settings | None = None) -> BoundarySet:
    """Read the admin2 GeoJSON into a registry frame plus shapely geometries."""
    from shapely.geometry import shape

    settings = settings or get_settings()
    path = download_boundaries(settings)
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload["features"]

    records: list[dict] = []
    geometries: dict[str, object] = {}
    for feature in features:
        properties = feature["properties"]
        code = properties["adm2_pcode"]
        state = _normalise_state(properties["adm1_name"])
        zone = GEOPOLITICAL_ZONES.get(state)
        if zone is None:
            raise ValueError(f"State {state!r} is not mapped to a geopolitical zone")
        geometry = shape(feature["geometry"])
        geometries[code] = geometry
        records.append(
            {
                "lga_code": code,
                "lga_name": properties["adm2_name"],
                "state_code": properties["adm1_pcode"],
                "state_name": state,
                "zone": zone,
                "senatorial_district": properties.get("sendist_en"),
                "area_sqkm": float(properties.get("area_sqkm") or 0.0),
                "centre_lat": float(properties["center_lat"]),
                "centre_lon": float(properties["center_lon"]),
            }
        )

    frame = pd.DataFrame.from_records(records).sort_values("lga_code").reset_index(drop=True)
    LOGGER.info(
        "loaded %d LGAs across %d states", len(frame), frame["state_name"].nunique()
    )
    return BoundarySet(frame=frame, geometries=geometries)


def _degrees_to_km(length_degrees: float, latitude: float) -> float:
    return length_degrees * EARTH_KM_PER_DEGREE * math.cos(math.radians(latitude)) ** 0.5


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def build_adjacency(
    boundaries: BoundarySet, settings: Settings | None = None
) -> pd.DataFrame:
    """Derive a weighted, row-normalised neighbour graph from shared borders."""
    from shapely.strtree import STRtree

    settings = settings or get_settings()
    reference_cfg = settings.section("reference")
    buffer_degrees = float(reference_cfg["adjacency_buffer_degrees"])
    min_neighbours = int(reference_cfg["adjacency_min_neighbours"])

    frame = boundaries.frame
    codes = frame["lga_code"].tolist()
    centres = {
        row.lga_code: (row.centre_lat, row.centre_lon) for row in frame.itertuples()
    }
    geometries = [boundaries.geometries[code] for code in codes]
    tree = STRtree(geometries)

    edges: list[dict] = []
    for index, code in enumerate(codes):
        geometry = geometries[index]
        # A small buffer catches units separated by sliver gaps in the source polygons.
        probe = geometry.buffer(buffer_degrees)
        for candidate_index in tree.query(probe):
            candidate_index = int(candidate_index)
            if candidate_index == index:
                continue
            other_code = codes[candidate_index]
            other = geometries[candidate_index]
            shared = geometry.buffer(buffer_degrees).intersection(other.buffer(buffer_degrees))
            if shared.is_empty:
                continue
            border_degrees = geometry.boundary.intersection(other.boundary).length
            if border_degrees == 0.0:
                # Touching only after buffering, treat as a weak border.
                border_degrees = buffer_degrees
            lat_a, lon_a = centres[code]
            lat_b, lon_b = centres[other_code]
            edges.append(
                {
                    "lga_code": code,
                    "neighbour_code": other_code,
                    "border_km": _degrees_to_km(border_degrees, (lat_a + lat_b) / 2),
                    "centroid_km": haversine_km(lat_a, lon_a, lat_b, lon_b),
                }
            )

    adjacency = pd.DataFrame.from_records(edges)
    adjacency = _fill_isolated(adjacency, frame, centres, min_neighbours)
    totals = adjacency.groupby("lga_code")["border_km"].transform("sum")
    adjacency["weight"] = (adjacency["border_km"] / totals).fillna(0.0)

    LOGGER.info(
        "adjacency graph: %d edges, mean degree %.2f",
        len(adjacency),
        len(adjacency) / max(len(frame), 1),
    )
    return adjacency[
        ["lga_code", "neighbour_code", "weight", "border_km", "centroid_km"]
    ].reset_index(drop=True)


def _fill_isolated(
    adjacency: pd.DataFrame,
    frame: pd.DataFrame,
    centres: dict[str, tuple[float, float]],
    min_neighbours: int,
) -> pd.DataFrame:
    """Give island and near-island LGAs their nearest units as weak neighbours.

    Riverine units in Bayelsa, Rivers and Lagos can share no land border at the
    resolution of the source polygons. Leaving them isolated would switch off the
    spatial term of the Hawkes kernel for exactly the areas where waterborne
    movement matters, so they receive distance-weighted nearest neighbours.
    """
    degree = adjacency.groupby("lga_code").size() if not adjacency.empty else pd.Series(dtype=int)
    additions: list[dict] = []
    for code in frame["lga_code"]:
        if int(degree.get(code, 0)) >= min_neighbours:
            continue
        lat, lon = centres[code]
        distances = sorted(
            (
                (haversine_km(lat, lon, other_lat, other_lon), other_code)
                for other_code, (other_lat, other_lon) in centres.items()
                if other_code != code
            )
        )[:min_neighbours]
        for distance_km, other_code in distances:
            additions.append(
                {
                    "lga_code": code,
                    "neighbour_code": other_code,
                    "border_km": max(1.0, 50.0 / max(distance_km, 1.0)),
                    "centroid_km": distance_km,
                }
            )
    if not additions:
        return adjacency
    extra = pd.DataFrame.from_records(additions)
    combined = pd.concat([adjacency, extra], ignore_index=True)
    return combined.drop_duplicates(subset=["lga_code", "neighbour_code"], keep="first")


def write_simplified_geojson(
    boundaries: BoundarySet,
    settings: Settings | None = None,
    tolerance_degrees: float = 0.01,
) -> Path:
    """Write a lightweight copy of the polygons for the map layer.

    The source file carries survey grade geometry, several thousand vertices for
    some units, which is the right thing for the point in polygon join and the
    wrong thing to push into a browser: the full collection is close to six
    megabytes and stalls the client before a single tier is visible. Simplifying
    to roughly a kilometre of tolerance keeps every boundary recognisable at
    national and state zoom, which is the only scale the dashboard uses, and cuts
    the payload by an order of magnitude. Analysis always reads the original.
    """
    settings = settings or get_settings()
    target = settings.paths.reference / "nga_admin2_simplified.geojson"
    features = []
    for row in boundaries.frame.itertuples():
        geometry = boundaries.geometries[row.lga_code].simplify(
            tolerance_degrees, preserve_topology=True
        )
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "adm2_pcode": row.lga_code,
                    "adm2_name": row.lga_name,
                    "adm1_name": row.state_name,
                },
                "geometry": geometry.__geo_interface__,
            }
        )
    payload = {"type": "FeatureCollection", "features": features}
    target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    LOGGER.info(
        "simplified boundaries written to %s (%.1f MB)",
        target.name,
        target.stat().st_size / 1e6,
    )
    return target


def assign_points_to_lgas(
    points: pd.DataFrame,
    boundaries: BoundarySet,
    lat_column: str = "latitude",
    lon_column: str = "longitude",
) -> pd.Series:
    """Map incident coordinates to LGA codes by point in polygon, then nearest centroid.

    Textual LGA fields in the event feeds are inconsistent in spelling and often
    empty, so the coordinates are treated as the authoritative locator. Points
    that fall outside every polygon, usually because of coarse geocoding to a
    town or a river, are snapped to the nearest centroid within 40 km and
    otherwise left unassigned rather than forced into a unit.
    """
    import numpy as np
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    codes = boundaries.frame["lga_code"].tolist()
    geometries = [boundaries.geometries[code] for code in codes]
    tree = STRtree(geometries)

    latitudes = pd.to_numeric(points[lat_column], errors="coerce").to_numpy(dtype=float)
    longitudes = pd.to_numeric(points[lon_column], errors="coerce").to_numpy(dtype=float)

    assignments: list[str | None] = [None] * len(points)
    unresolved: list[int] = []
    for position in range(len(points)):
        latitude, longitude = latitudes[position], longitudes[position]
        if not (np.isfinite(latitude) and np.isfinite(longitude)):
            continue
        point = Point(float(longitude), float(latitude))
        for candidate_index in tree.query(point):
            candidate_index = int(candidate_index)
            if geometries[candidate_index].contains(point):
                assignments[position] = codes[candidate_index]
                break
        else:
            unresolved.append(position)

    if unresolved:
        centre_lat = np.radians(boundaries.frame["centre_lat"].to_numpy(dtype=float))
        centre_lon = np.radians(boundaries.frame["centre_lon"].to_numpy(dtype=float))
        query_lat = np.radians(latitudes[unresolved])
        query_lon = np.radians(longitudes[unresolved])
        d_phi = centre_lat[None, :] - query_lat[:, None]
        d_lambda = centre_lon[None, :] - query_lon[:, None]
        haversine = (
            np.sin(d_phi / 2) ** 2
            + np.cos(query_lat)[:, None] * np.cos(centre_lat)[None, :] * np.sin(d_lambda / 2) ** 2
        )
        distances = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
        nearest = distances.argmin(axis=1)
        best = distances[np.arange(len(unresolved)), nearest]
        for offset, position in enumerate(unresolved):
            if best[offset] <= 40.0:
                assignments[position] = codes[int(nearest[offset])]

    resolved = sum(1 for value in assignments if value is not None)
    LOGGER.info(
        "located %d of %d points (%.1f%% inside a polygon, %d snapped to a centroid)",
        resolved,
        len(points),
        100.0 * (resolved - sum(1 for i in unresolved if assignments[i])) / max(len(points), 1),
        sum(1 for i in unresolved if assignments[i]),
    )
    return pd.Series(assignments, index=points.index, dtype="object")
