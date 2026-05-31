# s2vec-reproduction

## Paper
**S2Vec: Self-Supervised Geospatial Embeddings for the Built Environment**
Choudhury et al., Google Research, ACM TSAS 2026

## Architecture Summary

### Three-Stage Pipeline

**Stage 1 — Feature Vector Construction**
- Partition area into S2 level-12 cells (~5 km² each)
- For each cell, build histogram vector Θ(s_l) of:
  - PoI category counts (shops, restaurants, gas stations, etc.) — via OSM
  - Road network feature counts (roads, traffic lights, etc.) — via OSMnx
- Use raw counts (no normalization at this stage)

**Stage 2 — Rasterization**
- Pick parent level l' = 8 → each parent has 2^(12-8) × 2^(12-8) = 16×16 child cells
- Each parent cell → one "image" of shape [16, 16, F] where F = feature vector dim
- Row-by-row spatial ordering of children within parent (preserves geography)
- Dataset = one image per parent cell covering the area of interest

**Stage 3 — MAE Training + Embedding Extraction**
- Standard ViT-based MAE (He et al. 2022) on [16×16, F] "images"
- Each patch = one child cell's feature vector Θ(s_l) (patch grid = 1×1)
- Masking ratio: 75% (MAE default)
- After training: run patch encoder on each Θ(s_l) individually → Φ(s_l)
- Downstream evaluation: 2-layer MLP on frozen embeddings


## Implementation

### Download data

1. **Download PoIs** — `osmnx.features_from_place` for four OSM tag namespaces:
   - `amenity` (23 types: restaurant, cafe, school, hospital, fuel, …)
   - `shop` (13 types: supermarket, electronics, bakery, …)
   - `tourism` (5 types: hotel, museum, attraction, …)
   - `leisure` (6 types: park, sports_centre, pitch, …)

2. **Download road network** — `osmnx.graph_from_place` with `network_type="all"`, converted to node/edge GeoDataFrames.

3. **Assign S2 cells** — every feature's centroid is mapped to an S2 level-12 cell ID via `s2sphere`.

### Rasterize

Each S2 level-8 parent cell contains exactly 4^(12−8) = **256 level-12 children**, which tile a geographic rectangle in a 16×16 grid.

### Training

Model: MAE paper (An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)

### Interpretation

Scree plot — one axis dominates (concern)

PC1 explains 61% of variance, PC2 adds 12%, and 80% is reached by PC4. This is a warning sign: a 384-dimensional space that collapses to effectively 3–4 meaningful dimensions means the
attention layers aren't learning diverse features — the encoder is mostly doing a weighted sum of the raw count magnitudes. The ideal shape would be a gentler slope reaching 80% around
PC8–10.

PCA by price — three hard clusters, not a smooth manifold

The embedding space has split into three distinct groups rather than a continuous manifold

The embedding space has split into three distinct groups rather than a continuous manifold:

- Right cluster (PC1 ≈ 7–17): the main bulk of LA cells, with a clear price gradient from purple (bottom, ~$80k) to yellow (top-right, ~$500k+). This is good — within this cluster the model
has learned something real about neighbourhood structure.
- Bottom-center cluster (PC1 ≈ 3–5, PC2 ≈ −9): all high-priced yellow dots. Likely Westside / coastal cells that share a specific feature pattern (high restaurant/shop density, low
industrial).
- Left outlier cluster (PC1 ≈ −22): medium-high price, very far from everything else. Cells with low total OSM count but relatively high income — probably hillside residential areas (few
tagged POIs, quiet streets).

The hard separation between clusters is the real problem. It suggests the model is making a near-binary decision early in the encoder rather than interpolating smoothly.

Spatial plot

Need to double check whether the east–west gradient (red/eastern ↔ blue/western) is real.


