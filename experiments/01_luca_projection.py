#!/usr/bin/env python
# experiments/01_luca_projection.py
"""Experiment: Project PBMC cells onto LUCA subspace.

This script:
1. Loads human PBMC scRNA‑seq data.
2. Encodes each cell into a 64‑state SUBIT distribution.
3. Computes the LUCA mass (probability that the cell lies in one of the 18 LUCA states).
4. Visualises the distribution of LUCA mass and its relationship to cell cycle phase.
5. Saves figures to results/luca_projection/.
"""

import sys
import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# Add repository root to path (if running from subdirectory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.human import load_human_pbmc_with_cycle_scores
from encoder.universal import UniversalEncoder
from metrics.luca import luca_projection
from levels.level0_luca import Level0_LUCA


def main():
    # Create output directory
    out_dir = "results/luca_projection"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load data
    print("Loading PBMC data with cell cycle scores...")
    adata = load_human_pbmc_with_cycle_scores()
    print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")

    # 2. Encode to SUBIT
    print("Encoding to SUBIT 64‑state distributions...")
    encoder = UniversalEncoder(normalize=True)  # applies log1p normalisation
    P64 = encoder.encode(adata)
    adata.obsm["X_subit64"] = P64

    # 3. Compute LUCA mass
    luca_level = Level0_LUCA()
    allowed_luca_states = luca_level.allowed_states()
    luca_mass = luca_projection(P64, allowed_luca_states)
    adata.obs["luca_mass"] = luca_mass
    print(f"LUCA mass: mean = {np.mean(luca_mass):.4f}, std = {np.std(luca_mass):.4f}")

    # 4. Visualisations
    sns.set_style("whitegrid")
    
    # Histogram of LUCA mass
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(luca_mass, bins=30, kde=True, ax=ax)
    ax.set_xlabel("LUCA mass (probability in LUCA states)")
    ax.set_ylabel("Number of cells")
    ax.set_title("Distribution of LUCA mass across PBMC cells")
    fig.savefig(os.path.join(out_dir, "luca_mass_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # LUCA mass by cell cycle phase (boxplot)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="phase", y="luca_mass", data=adata.obs, ax=ax)
    ax.set_ylabel("LUCA mass")
    ax.set_title("LUCA mass by cell cycle phase")
    fig.savefig(os.path.join(out_dir, "luca_mass_by_phase.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. UMAP of SUBIT space coloured by LUCA mass
    print("Computing UMAP in SUBIT space...")
    sc.pp.neighbors(adata, use_rep="X_subit64", n_pcs=None)  # use full 64D space
    sc.tl.umap(adata, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(adata, color="luca_mass", ax=ax, show=False, title="LUCA mass (SUBIT space)")
    fig.savefig(os.path.join(out_dir, "umap_luca_mass.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also colour by cell cycle phase for reference
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(adata, color="phase", ax=ax, show=False, title="Cell cycle phase (SUBIT space)")
    fig.savefig(os.path.join(out_dir, "umap_phase.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"All figures saved to {out_dir}/")
    print("Experiment completed.")


if __name__ == "__main__":
    main()