#!/usr/bin/env python3
"""
feature_visualize.py — Stage 1 QA visualizations for features.parquet.

Outputs (written to plots/)
---------------------------
feature_sparsity.png      — fraction of cells with zero count, per feature
feature_distributions.png — non-zero count histograms (log-scale) per feature
cell_density.png          — histogram of total OSM count-sum per cell
feature_correlation.png   — Pearson correlation heatmap across features
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURES_PATH = Path("../data/processed/features.parquet")
PLOTS_DIR = Path("plots")

# Feature group prefixes for grouping in correlation heatmap
GROUPS = ["amenity", "shop", "tourism", "leisure", "road"]


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"{FEATURES_PATH} not found — run download_data.py first")
    df = pd.read_parquet(FEATURES_PATH)
    log.info("Loaded %d cells × %d features", *df.shape)
    return df


# ── Plot 1: Sparsity ───────────────────────────────────────────────────────────

def plot_sparsity(df: pd.DataFrame) -> None:
    sparsity = (df == 0).mean().sort_values(ascending=False)

    n = len(sparsity)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.22), 5))
    colors = [
        "#4e79a7" if c.startswith("road") else "#f28e2b"
        for c in sparsity.index
    ]
    ax.bar(range(n), sparsity.values, color=colors, width=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(sparsity.index, rotation=90, fontsize=7)
    ax.set_ylabel("Fraction of cells with zero count")
    ax.set_title("Feature sparsity (fraction of zero-value cells per feature)")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#4e79a7", label="road"), Patch(color="#f28e2b", label="PoI")],
        fontsize=8,
    )
    fig.tight_layout()
    out = PLOTS_DIR / "feature_sparsity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


# ── Plot 2: Per-feature count distributions ────────────────────────────────────

def plot_distributions(df: pd.DataFrame) -> None:
    cols = df.columns.tolist()
    n = len(cols)
    ncols = 8
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.0))
    axes_flat = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        vals = df[col][df[col] > 0]
        if len(vals) == 0:
            ax.text(0.5, 0.5, "all zero", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="gray")
        else:
            ax.hist(vals, bins=30, log=True, color="#4e79a7", edgecolor="none")
            ax.set_xlabel("count", fontsize=6)
        ax.set_title(col, fontsize=6, pad=2)
        ax.tick_params(labelsize=5)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Non-zero count distributions per feature (log y-axis)", fontsize=10, y=1.01)
    fig.tight_layout()
    out = PLOTS_DIR / "feature_distributions.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# ── Plot 3: Cell density ───────────────────────────────────────────────────────

def plot_cell_density(df: pd.DataFrame) -> None:
    totals = df.sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(totals, bins=60, color="#59a14f", edgecolor="none")
    axes[0].set_xlabel("Total OSM feature count per cell")
    axes[0].set_ylabel("Number of cells")
    axes[0].set_title("Cell density (linear)")

    nonzero = totals[totals > 0]
    axes[1].hist(np.log1p(nonzero), bins=60, color="#59a14f", edgecolor="none")
    axes[1].set_xlabel("log(1 + total count)")
    axes[1].set_ylabel("Number of cells")
    axes[1].set_title(f"Cell density (log scale, n={len(nonzero)} non-empty)")

    fig.suptitle("Distribution of total OSM feature counts per S2 level-12 cell")
    fig.tight_layout()
    out = PLOTS_DIR / "cell_density.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


# ── Plot 4: Correlation heatmap ────────────────────────────────────────────────

def plot_correlation(df: pd.DataFrame) -> None:
    # Sort columns by group prefix for a readable block structure
    def group_key(name: str) -> tuple[int, str]:
        for i, g in enumerate(GROUPS):
            if name.startswith(g):
                return (i, name)
        return (len(GROUPS), name)

    sorted_cols = sorted(df.columns, key=group_key)
    corr = df[sorted_cols].corr(method="pearson")

    n = len(sorted_cols)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.28), max(8, n * 0.26)))
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)         # hide the diagonal (always 1.0)
    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0,
        xticklabels=sorted_cols,
        yticklabels=sorted_cols,
        cbar_kws={"shrink": 0.6},
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=5, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5, rotation=0)
    ax.set_title("Feature–feature Pearson correlation (grouped by OSM tag namespace)")
    fig.tight_layout()
    out = PLOTS_DIR / "feature_correlation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# ── Summary stats ──────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    totals = df.sum(axis=1)
    print(f"\n{'─'*50}")
    print(f"  Cells:         {len(df):>8,}")
    print(f"  Features (F):  {df.shape[1]:>8,}")
    print(f"  Non-empty cells: {(totals > 0).sum():>6,}  ({(totals > 0).mean():.1%})")
    print(f"  Mean count/cell: {totals.mean():>8.1f}")
    print(f"  Median count/cell: {totals.median():>6.1f}")
    sparsity = (df == 0).mean()
    print(f"  Mean feature sparsity: {sparsity.mean():.1%}")
    top5_sparse = sparsity.sort_values(ascending=False).head(5)
    print(f"  5 sparsest features:")
    for name, val in top5_sparse.items():
        print(f"    {name:<35} {val:.1%} zeros")
    print(f"{'─'*50}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features()
    print_summary(df)
    plot_sparsity(df)
    plot_distributions(df)
    plot_cell_density(df)
    plot_correlation(df)
    log.info("All plots written to %s/", PLOTS_DIR)


if __name__ == "__main__":
    main()
