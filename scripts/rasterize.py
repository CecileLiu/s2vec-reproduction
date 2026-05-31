#!/usr/bin/env python3
"""
rasterize.py — Stage 2: Arrange S2 level-12 cells into 16×16 images.

For each S2 level-8 parent that has at least MIN_FILLED non-empty
level-12 children in features.parquet, builds a 16×16 geographic grid
and fills each position with the child's feature vector (zero-padded
where no OSM data exists).

Run from the scripts/ directory:
    python rasterize.py

Outputs (data/processed/)
--------------------------
images.npy          — float32 [N, 16, 16, F]   raw count tensors
image_cell_ids.npy  — int64   [N, 16, 16]      S2 level-12 cell ID per grid pos
parent_cell_ids.npy — int64   [N]              S2 level-8 parent cell ID per image
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import s2sphere
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("../data/processed")
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

S2_PATCH_LEVEL = 12
S2_IMAGE_LEVEL = 8
GRID_SIZE = 16       # 2^(12-8) children per side
MIN_FILLED = 1       # min non-zero children to include a parent image


# ── S2 helpers ─────────────────────────────────────────────────────────────────

def _mask64(val: int) -> int:
    """Mask to 64 bits → always a non-negative Python int, safe for s2sphere.

    s2sphere is pure Python and stores cell IDs as arbitrary-precision ints.
    Passing a negative value (e.g. a signed int64 read back from parquet)
    breaks s2sphere's internal bit operations.  This mask restores the correct
    unsigned 64-bit representation before any s2sphere call.
    """
    return val & 0xFFFF_FFFF_FFFF_FFFF


def _to_signed(val: int) -> int:
    """Unsigned 64-bit cell ID → signed int64 for numpy int64 storage."""
    val = _mask64(val)
    return val if val < (1 << 63) else val - (1 << 64)


def cell_parent(cell_id: int, level: int = S2_IMAGE_LEVEL) -> int:
    """Return the level-`level` parent ID as an unsigned Python int."""
    return s2sphere.CellId(_mask64(cell_id)).parent(level).id()


def cell_centroid_deg(cell_id: int) -> tuple[float, float]:
    """Return (lat_deg, lon_deg) for the centroid of an S2 cell."""
    ll = s2sphere.CellId(_mask64(cell_id)).to_lat_lng()
    return ll.lat().degrees, ll.lng().degrees


def get_patch_children(parent_id: int) -> list[int]:
    """Return all S2_PATCH_LEVEL children as unsigned Python ints (256 for 8→12)."""
    root = s2sphere.CellId(_mask64(parent_id))

    def descend(cid: s2sphere.CellId, target: int) -> list[int]:
        if cid.level() == target:
            return [cid.id()]          # raw unsigned int — safe for s2sphere
        result: list[int] = []
        for pos in range(4):
            result.extend(descend(cid.child(pos), target))
        return result

    children = descend(root, S2_PATCH_LEVEL)
    assert len(children) == GRID_SIZE * GRID_SIZE, (
        f"Expected {GRID_SIZE**2} children, got {len(children)}"
    )
    return children


# ── Grid-position mapping ───────────────────────────────────────────────────────

def assign_grid_positions(children: list[int]) -> dict[int, tuple[int, int]]:
    """Map each child cell ID to a (row, col) in the 16×16 grid.

    Row 0 = northernmost, row 15 = southernmost  (N→S).
    Col 0 = westernmost,  col 15 = easternmost   (W→E).

    Uses the centroid bounding-box rounding formula from the implementation
    guide.  Verifies no two children map to the same position.
    """
    centroids = [cell_centroid_deg(cid) for cid in children]
    lats = [c[0] for c in centroids]
    lons = [c[1] for c in centroids]

    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    lat_span = lat_max - lat_min or 1.0   # guard against degenerate cell
    lon_span = lon_max - lon_min or 1.0

    mapping: dict[int, tuple[int, int]] = {}
    for cid, (lat, lon) in zip(children, centroids):
        row = int(round((lat_max - lat) / lat_span * (GRID_SIZE - 1)))
        col = int(round((lon - lon_min) / lon_span * (GRID_SIZE - 1)))
        row = max(0, min(GRID_SIZE - 1, row))
        col = max(0, min(GRID_SIZE - 1, col))
        mapping[cid] = (row, col)

    occupied = list(mapping.values())
    if len(set(occupied)) != len(occupied):
        # Fall back: rank-based assignment (sort by lat desc then lon asc)
        mapping = _rank_based_positions(children, centroids)

    return mapping


def _rank_based_positions(
    children: list[int], centroids: list[tuple[float, float]]
) -> dict[int, tuple[int, int]]:
    """Fallback: sort all 256 children by (lat desc, lon asc), assign row-major."""
    order = sorted(
        range(len(children)),
        key=lambda i: (-centroids[i][0], centroids[i][1]),
    )
    mapping: dict[int, tuple[int, int]] = {}
    for linear_idx, child_idx in enumerate(order):
        row = linear_idx // GRID_SIZE
        col = linear_idx % GRID_SIZE
        mapping[children[child_idx]] = (row, col)
    return mapping


# ── Rasterization ───────────────────────────────────────────────────────────────

def rasterize(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    F = features.shape[1]
    feat_np = features.to_numpy(dtype=np.float32)
    # Keys are unsigned ints throughout; parquet may have stored IDs as signed int64
    idx_lookup: dict[int, int] = {_mask64(int(cid)): i for i, cid in enumerate(features.index)}

    # 1. Map each patch cell to its level-8 parent
    log.info("Computing level-%d parents for %d cells ...", S2_IMAGE_LEVEL, len(features))
    parent_to_cells: dict[int, list[int]] = defaultdict(list)
    for cid in features.index:
        unsigned_cid = _mask64(int(cid))
        parent_to_cells[cell_parent(unsigned_cid)].append(unsigned_cid)

    # 2. Filter parents by minimum fill
    valid_parents = sorted(
        pid for pid, cells in parent_to_cells.items()
        if sum(1 for c in cells if feat_np[idx_lookup[c]].any()) >= MIN_FILLED
    )
    log.info(
        "  %d / %d parents have ≥%d non-empty children",
        len(valid_parents), len(parent_to_cells), MIN_FILLED,
    )

    # 3. Build output arrays
    N = len(valid_parents)
    images = np.zeros((N, GRID_SIZE, GRID_SIZE, F), dtype=np.float32)
    image_cell_ids = np.zeros((N, GRID_SIZE, GRID_SIZE), dtype=np.int64)
    # Convert unsigned parent IDs to signed int64 only at numpy write time
    parent_cell_ids = np.array([_to_signed(p) for p in valid_parents], dtype=np.int64)

    fallback_count = 0
    for img_idx, pid in enumerate(tqdm(valid_parents, desc="Rasterizing")):
        children = get_patch_children(pid)   # returns unsigned ints
        grid_map = assign_grid_positions(children)

        # Detect if fallback was used (rounding formula always returns GRID_SIZE² unique pos)
        if len(set(grid_map.values())) != GRID_SIZE * GRID_SIZE:
            fallback_count += 1

        for cid, (row, col) in grid_map.items():
            image_cell_ids[img_idx, row, col] = _to_signed(cid)
            if cid in idx_lookup:
                images[img_idx, row, col] = feat_np[idx_lookup[cid]]

    if fallback_count:
        log.warning(
            "%d parents used rank-based fallback positioning", fallback_count
        )

    return images, image_cell_ids, parent_cell_ids


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found — run download_data.py first"
        )

    log.info("Loading %s ...", FEATURES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    log.info("  %d cells × %d features", *features.shape)

    images, image_cell_ids, parent_cell_ids = rasterize(features)

    np.save(PROCESSED_DIR / "images.npy", images)
    np.save(PROCESSED_DIR / "image_cell_ids.npy", image_cell_ids)
    np.save(PROCESSED_DIR / "parent_cell_ids.npy", parent_cell_ids)

    log.info(
        "images.npy          shape=%s  dtype=%s", images.shape, images.dtype
    )
    log.info(
        "image_cell_ids.npy  shape=%s  dtype=%s",
        image_cell_ids.shape, image_cell_ids.dtype,
    )
    log.info(
        "parent_cell_ids.npy shape=%s  dtype=%s",
        parent_cell_ids.shape, parent_cell_ids.dtype,
    )

    filled = (images.sum(axis=-1) > 0).mean()
    log.info("Grid fill rate: %.1f%% of positions have non-zero features", filled * 100)


if __name__ == "__main__":
    main()
