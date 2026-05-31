#!/usr/bin/env python3
"""
download_data.py — Stage 1: OSM data download for S2Vec.

Downloads PoI and road-network data for Los Angeles via OSMnx,
assigns each feature to an S2 level-12 cell, and writes per-cell
raw count vectors to data/processed/features.parquet.

Outputs
-------
data/raw/pois.parquet          — cached raw PoI geometries + categories
data/raw/road_nodes.parquet    — cached OSMnx nodes
data/raw/road_edges.parquet    — cached OSMnx edges
data/processed/features.parquet       — cell_id × feature count matrix (int32)
data/processed/feature_columns.json   — ordered list of column names
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import s2sphere

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

PLACE = "Los Angeles, California, USA"
S2_PATCH_LEVEL = 12   # one feature vector per cell (~5 km²)
S2_IMAGE_LEVEL = 8    # parent level; each parent has 16×16 children

DATA_DIR = Path(__file__).parents[1]/ "data" # Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# OSM PoI tags to collect.  key → list of values to keep.
POI_TAGS: dict[str, list[str]] = {
    "amenity": [
        "restaurant", "cafe", "fast_food", "bar", "pub",
        "bank", "atm", "pharmacy", "hospital", "clinic",
        "school", "university", "kindergarten",
        "fuel", "parking", "post_office", "police",
        "fire_station", "library", "cinema", "theatre",
        "place_of_worship", "dentist", "doctors",
    ],
    "shop": [
        "supermarket", "convenience", "clothes", "bakery",
        "butcher", "electronics", "hardware", "hairdresser",
        "beauty", "florist", "furniture", "books", "alcohol",
    ],
    "tourism": ["hotel", "museum", "attraction", "viewpoint", "guest_house"],
    "leisure": [
        "park", "sports_centre", "pitch", "playground",
        "golf_course", "fitness_centre",
    ],
}

# Road segment highway types to track.
HIGHWAY_TYPES = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "service", "footway", "cycleway",
    "path", "unclassified",
]


# ── S2 helpers ─────────────────────────────────────────────────────────────────

def latlon_to_cell_id(lat: float, lon: float, level: int = S2_PATCH_LEVEL) -> int:
    ll = s2sphere.LatLng.from_degrees(lat, lon)
    return s2sphere.CellId.from_lat_lng(ll).parent(level).id()


def add_cell_id(gdf: gpd.GeoDataFrame, level: int = S2_PATCH_LEVEL) -> gpd.GeoDataFrame:
    """Append 'cell_id' by mapping each geometry's centroid to an S2 cell."""
    centroids = gdf.geometry.centroid
    gdf = gdf.copy()
    gdf["cell_id"] = [latlon_to_cell_id(p.y, p.x, level) for p in centroids]
    return gdf


# ── Download (with disk caching) ───────────────────────────────────────────────

def download_pois(place: str) -> gpd.GeoDataFrame:
    cache = RAW_DIR / "pois.parquet"
    if cache.exists():
        log.info("Loading cached PoIs from %s", cache)
        return gpd.read_parquet(cache)

    log.info("Downloading PoIs for '%s' ...", place)
    frames: list[gpd.GeoDataFrame] = []
    for key, values in POI_TAGS.items():
        try:
            gdf = ox.features_from_place(place, tags={key: values})
        except Exception as exc:
            log.warning("  %s: skipped (%s)", key, exc)
            continue
        if key not in gdf.columns:
            continue
        sub = gdf[["geometry", key]].copy()
        sub = sub[sub[key].isin(values)]          # drop unmapped tag values
        sub["category"] = key + "_" + sub[key].astype(str)
        frames.append(sub[["geometry", "category"]])
        log.info("  %s: %d features", key, len(sub))

    out = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), crs="EPSG:4326"
    )
    out.to_parquet(cache)
    log.info("Saved raw PoIs → %s  (%d rows)", cache, len(out))
    return out


def download_road_network(place: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes_cache = RAW_DIR / "road_nodes.parquet"
    edges_cache = RAW_DIR / "road_edges.parquet"

    if nodes_cache.exists() and edges_cache.exists():
        log.info("Loading cached road network")
        return gpd.read_parquet(nodes_cache), gpd.read_parquet(edges_cache)

    log.info("Downloading road network for '%s' ...", place)
    G = ox.graph_from_place(place, network_type="all", retain_all=False, simplify=False)
    nodes, edges = ox.graph_to_gdfs(G)
    nodes = nodes.to_crs("EPSG:4326")
    edges = edges.to_crs("EPSG:4326")
    nodes.to_parquet(nodes_cache)
    edges.to_parquet(edges_cache)
    log.info("  Nodes: %d  Edges: %d", len(nodes), len(edges))
    return nodes, edges


# ── Feature vector construction ────────────────────────────────────────────────

def poi_feature_columns() -> list[str]:
    return [f"{key}_{v}" for key, values in POI_TAGS.items() for v in values]


def build_poi_counts(
    pois: gpd.GeoDataFrame, cell_ids: list[int]
) -> pd.DataFrame:
    """Count PoIs per (cell, category) pair; rows = cell_ids, cols = category names."""
    valid = set(poi_feature_columns())
    pois = pois[pois["category"].isin(valid)]
    counts = (
        pois.groupby(["cell_id", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=cell_ids, columns=poi_feature_columns(), fill_value=0)
    )
    return counts


def build_road_counts(
    edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, cell_ids: list[int]
) -> pd.DataFrame:
    """Count road segments by type, traffic signals, and intersections per cell."""

    def normalize_hw(val) -> str:
        if isinstance(val, list):
            val = val[0]
        s = str(val)
        return s if s in HIGHWAY_TYPES else "other"

    edges = edges.copy()
    edges["hw"] = edges["highway"].apply(normalize_hw)
    hw_cols = [f"road_{h}" for h in HIGHWAY_TYPES + ["other"]]

    road_counts = (
        edges.groupby(["cell_id", "hw"])
        .size()
        .unstack(fill_value=0)
        .rename(columns=lambda c: f"road_{c}")
        .reindex(index=cell_ids, columns=hw_cols, fill_value=0)
    )

    # Traffic signals: nodes tagged highway=traffic_signals
    signal_mask = nodes.get("highway", pd.Series(dtype=str)) == "traffic_signals"
    signal_counts = (
        nodes[signal_mask]
        .groupby("cell_id")
        .size()
        .reindex(cell_ids, fill_value=0)
        .rename("road_traffic_signals")
        .to_frame()
    )

    # Intersections: nodes where ≥3 roads meet
    if "street_count" in nodes.columns:
        inter_counts = (
            nodes[nodes["street_count"] >= 3]
            .groupby("cell_id")
            .size()
            .reindex(cell_ids, fill_value=0)
            .rename("road_intersections")
            .to_frame()
        )
    else:
        inter_counts = pd.DataFrame(
            {"road_intersections": np.zeros(len(cell_ids), dtype=np.int32)},
            index=cell_ids,
        )

    return pd.concat([road_counts, signal_counts, inter_counts], axis=1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download raw data (cached after first run)
    pois = download_pois(PLACE)
    nodes, edges = download_road_network(PLACE)

    # 2. Assign every feature to an S2 level-12 cell via centroid
    log.info("Assigning S2 level-%d cells ...", S2_PATCH_LEVEL)
    pois = add_cell_id(pois)
    edges = add_cell_id(edges)
    nodes = add_cell_id(nodes)

    all_cell_ids = sorted(
        set(pois["cell_id"]) | set(edges["cell_id"]) | set(nodes["cell_id"])
    )
    log.info("  %d unique S2 cells", len(all_cell_ids))

    # 3. Build count feature vectors
    log.info("Building PoI count vectors ...")
    poi_counts = build_poi_counts(pois, all_cell_ids)

    log.info("Building road count vectors ...")
    road_counts = build_road_counts(edges, nodes, all_cell_ids)

    # 4. Combine and save
    features = (
        pd.concat([poi_counts, road_counts], axis=1)
        .fillna(0)
        .astype(np.int32)
    )
    features.index.name = "cell_id"

    feat_path = PROCESSED_DIR / "features.parquet"
    features.to_parquet(feat_path)
    log.info(
        "Saved feature matrix: %d cells × %d features → %s",
        len(features), features.shape[1], feat_path,
    )

    col_path = PROCESSED_DIR / "feature_columns.json"
    col_path.write_text(json.dumps(features.columns.tolist()))
    log.info("Feature column names → %s", col_path)


if __name__ == "__main__":
    main()
