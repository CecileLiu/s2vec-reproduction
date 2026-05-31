#!/usr/bin/env python3
"""
embedding_visualization.py — Extract MAE patch embeddings and visualise
against California housing prices.

Steps
-----
1. Load MAE encoder from checkpoint; run encode_no_mask on every image.
2. Map token outputs back to their S2 level-12 cell IDs.
3. Assign housing-price records to the same S2 cells.
4. PCA to 2D; optionally UMAP if umap-learn is installed.
5. Write plots to plots/embeddings/.

Run from the scripts/ directory:
    python embedding_visualization.py

Saved artefacts
---------------
../data/processed/embeddings.npy          [M, 384] float32  patch embeddings
../data/processed/embedding_cell_ids.npy  [M]      int64    matching cell IDs
plots/embeddings/pca_by_price.png
plots/embeddings/pca_spatial.png
plots/embeddings/pca_scree.png
plots/embeddings/umap_by_price.png        (only if umap-learn installed)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s2sphere
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── import MAE from sibling script ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from training import ENCODER_DIM, GRID_SIZE, MAE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("../data/processed")
CKPT_PATH     = Path("../checkpoints/mae_best.pt")
HOUSING_CSV   = Path("../data/validation/california_housing_price.csv")
PLOTS_DIR     = Path("plots/embeddings")

S2_PATCH_LEVEL = 12

# LA bounding box for spatial filtering
LA_LAT = (33.60, 34.35)
LA_LON = (-118.75, -117.85)


# ── S2 helpers ─────────────────────────────────────────────────────────────────

def _mask64(v: int) -> int:
    return v & 0xFFFF_FFFF_FFFF_FFFF

def _to_signed(v: int) -> int:
    v = _mask64(v)
    return v if v < (1 << 63) else v - (1 << 64)

def latlon_to_cell_id(lat: float, lon: float, level: int = S2_PATCH_LEVEL) -> int:
    ll = s2sphere.LatLng.from_degrees(lat, lon)
    return _to_signed(s2sphere.CellId.from_lat_lng(ll).parent(level).id())

def cell_id_to_latlon(cell_id: int) -> tuple[float, float]:
    ll = s2sphere.CellId(_mask64(cell_id)).to_lat_lng()
    return ll.lat().degrees, ll.lng().degrees


# ── Embedding extraction ────────────────────────────────────────────────────────

def extract_embeddings(device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run encode_no_mask on all images; return (embeddings, cell_ids)."""
    # Load checkpoint
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"{CKPT_PATH} not found — run training.py first"
        )
    ckpt = torch.load(CKPT_PATH, map_location=device)
    F = ckpt["feature_dim"]

    model = MAE(feature_dim=F).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Loaded MAE checkpoint (epoch %d, val_loss=%.4f)", ckpt["epoch"], ckpt["val_loss"])

    # Load images + norm stats
    images = np.load(PROCESSED_DIR / "images.npy")          # [N, 16, 16, F]
    cell_ids_grid = np.load(PROCESSED_DIR / "image_cell_ids.npy")  # [N, 16, 16]
    stats = np.load(PROCESSED_DIR / "norm_stats.npz")
    mean, std = stats["mean"], stats["std"]

    N = len(images)
    all_embeddings: list[np.ndarray] = []
    all_cell_ids: list[int] = []

    log.info("Extracting embeddings from %d images ...", N)
    with torch.no_grad():
        for i in range(N):
            img_norm = (images[i] - mean) / (std + 1e-6)    # [16, 16, F]
            x = torch.from_numpy(
                img_norm.reshape(1, GRID_SIZE * GRID_SIZE, F).astype(np.float32)
            ).to(device)
            emb = model.encode_no_mask(x)                   # [1, 256, 384]
            emb_np = emb.squeeze(0).cpu().numpy()           # [256, 384]

            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    cid = int(cell_ids_grid[i, row, col])
                    if cid != 0:
                        all_embeddings.append(emb_np[row * GRID_SIZE + col])
                        all_cell_ids.append(cid)

    embeddings  = np.array(all_embeddings, dtype=np.float32)   # [M, 384]
    cell_id_arr = np.array(all_cell_ids,   dtype=np.int64)     # [M]

    np.save(PROCESSED_DIR / "embeddings.npy",         embeddings)
    np.save(PROCESSED_DIR / "embedding_cell_ids.npy", cell_id_arr)
    log.info("Saved %d patch embeddings → %s", len(embeddings), PROCESSED_DIR)
    return embeddings, cell_id_arr


# ── Housing price join ─────────────────────────────────────────────────────────

def load_housing_with_cells() -> pd.DataFrame:
    """Load CSV, assign S2 level-12 cell IDs, filter to LA."""
    df = pd.read_csv(HOUSING_CSV).dropna(subset=["latitude", "longitude", "median_house_value"])

    # Filter to LA metro area
    mask = (
        (df["latitude"]  >= LA_LAT[0]) & (df["latitude"]  <= LA_LAT[1]) &
        (df["longitude"] >= LA_LON[0]) & (df["longitude"] <= LA_LON[1])
    )
    df = df[mask].copy()
    log.info("LA housing records: %d", len(df))

    df["cell_id"] = df.apply(
        lambda r: latlon_to_cell_id(r["latitude"], r["longitude"]), axis=1
    )
    return df


def join_embeddings_housing(
    embeddings: np.ndarray,
    cell_ids: np.ndarray,
    housing: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate multiple housing records per cell, join with embeddings."""
    # Median price per cell
    price_per_cell = (
        housing.groupby("cell_id")["median_house_value"]
        .median()
        .rename("price")
    )

    emb_df = pd.DataFrame(
        embeddings,
        index=pd.Index(cell_ids, name="cell_id"),
        columns=[f"e{i}" for i in range(embeddings.shape[1])],
    )
    joined = emb_df.join(price_per_cell, how="inner")
    log.info("Cells with housing data: %d", len(joined))
    return joined


# ── PCA ────────────────────────────────────────────────────────────────────────

def run_pca(embeddings: np.ndarray, n_components: int = 50) -> tuple[np.ndarray, PCA]:
    scaler = StandardScaler()
    Z = scaler.fit_transform(embeddings)
    pca = PCA(n_components=min(n_components, embeddings.shape[1]), random_state=42)
    coords = pca.fit_transform(Z)
    return coords, pca


# ── Plots ───────────────────────────────────────────────────────────────────────

def _colorbar(fig, ax, sc, label: str) -> None:
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(label, fontsize=9)


def plot_pca_by_price(joined: pd.DataFrame, pca_coords: np.ndarray) -> None:
    log_price = np.log1p(joined["price"].values)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=log_price, cmap="plasma", s=18, alpha=0.75, linewidths=0,
    )
    _colorbar(fig, ax, sc, "log(1 + median house value $)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("MAE patch embeddings — PCA coloured by housing price (LA)")
    fig.tight_layout()
    out = PLOTS_DIR / "pca_by_price.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_pca_spatial(joined: pd.DataFrame, pca_coords: np.ndarray) -> None:
    latlons = [cell_id_to_latlon(cid) for cid in joined.index]
    lats = np.array([ll[0] for ll in latlons])
    lons = np.array([ll[1] for ll in latlons])
    pc1 = pca_coords[:, 0]

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(lons, lats, c=pc1, cmap="RdBu_r", s=18, alpha=0.8, linewidths=0)
    _colorbar(fig, ax, sc, "PC 1")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial distribution of PC 1 (LA block groups)")
    ax.set_aspect("equal")
    fig.tight_layout()
    out = PLOTS_DIR / "pca_spatial.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_scree(pca: PCA) -> None:
    ev = pca.explained_variance_ratio_ * 100
    cumev = np.cumsum(ev)
    n = min(20, len(ev))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, n + 1), ev[:n], color="#4e79a7", label="Individual")
    ax.plot(range(1, n + 1), cumev[:n], "o-", color="#f28e2b", label="Cumulative")
    ax.axhline(80, color="gray", linewidth=0.8, linestyle="--", label="80% line")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA scree plot — MAE patch embeddings")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = PLOTS_DIR / "pca_scree.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_umap_by_price(joined: pd.DataFrame, embeddings_subset: np.ndarray) -> None:
    try:
        import umap  # noqa: PLC0415
    except ImportError:
        log.info("umap-learn not installed — skipping UMAP plot")
        return

    log.info("Running UMAP ...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    scaler = StandardScaler()
    coords = reducer.fit_transform(scaler.fit_transform(embeddings_subset))

    log_price = np.log1p(joined["price"].values)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=log_price, cmap="plasma", s=18, alpha=0.75, linewidths=0,
    )
    _colorbar(fig, ax, sc, "log(1 + median house value $)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("MAE patch embeddings — UMAP coloured by housing price (LA)")
    fig.tight_layout()
    out = PLOTS_DIR / "umap_by_price.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Extract (or load cached) embeddings
    emb_path = PROCESSED_DIR / "embeddings.npy"
    cid_path  = PROCESSED_DIR / "embedding_cell_ids.npy"
    if emb_path.exists() and cid_path.exists():
        log.info("Loading cached embeddings ...")
        embeddings = np.load(emb_path)
        cell_ids   = np.load(cid_path)
    else:
        embeddings, cell_ids = extract_embeddings(device)

    log.info("Embeddings: %s  dtype=%s", embeddings.shape, embeddings.dtype)

    # 2. Load housing data + join
    housing = load_housing_with_cells()
    joined  = join_embeddings_housing(embeddings, cell_ids, housing)

    if len(joined) == 0:
        log.warning("No overlap between embeddings and housing cells — check lat/lon bounds")
        return

    # 3. PCA on the joined subset (cells with housing data)
    emb_subset = joined[[f"e{i}" for i in range(embeddings.shape[1])]].values
    pca_coords, pca = run_pca(emb_subset, n_components=50)
    log.info(
        "Top-2 PCs explain %.1f%% of variance",
        pca.explained_variance_ratio_[:2].sum() * 100,
    )

    # 4. Plots
    plot_pca_by_price(joined, pca_coords)
    plot_pca_spatial(joined, pca_coords)
    plot_scree(pca)
    plot_umap_by_price(joined, emb_subset)

    log.info("All plots → %s/", PLOTS_DIR)


if __name__ == "__main__":
    main()
