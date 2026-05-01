# datasets/bacteria.py
"""Load bacterial (E. coli) single‑cell expression data."""

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

def load_ecoli_synthetic(n_cells=200, n_genes=1000, seed=42):
    """
    Generate synthetic E. coli expression data for testing.
    Real data should replace this in production.
    """
    np.random.seed(seed)
    # Create gene names like eco00001, eco00002, ... plus some functional names
    gene_names = [f"eco{i:05d}" for i in range(n_genes)]
    # Add a few known functional genes (for pattern matching)
    functional_genes = [
        "rplA", "rpsB", "rpmC",           # ribosome
        "atpA", "atpD", "ndh",            # energy
        "secY", "secE", "ompA",           # membrane
        "dnaA", "dnaN", "recA",           # replication
        "cheA", "cheY", "tsr",            # chemotaxis / signalling
        "lon", "clpP", "groEL"            # stress
    ]
    gene_names[:len(functional_genes)] = functional_genes
    
    # Simulate expression: log-normal with varying means per functional group
    X = np.random.lognormal(mean=0, sigma=1, size=(n_cells, n_genes))
    # Add structure: upregulate certain groups in different cells
    for i, cell in enumerate(range(n_cells)):
        if i < n_cells//3:
            # stress mode: high "lon", "clpP"
            for g in ["lon", "clpP", "groEL"]:
                if g in gene_names:
                    idx = gene_names.index(g)
                    X[i, idx] *= 5
        elif i < 2*n_cells//3:
            # growth mode: high ribosome genes
            for g in ["rplA", "rpsB", "rpmC"]:
                if g in gene_names:
                    idx = gene_names.index(g)
                    X[i, idx] *= 4
        else:
            # replication mode: high dnaA, recA
            for g in ["dnaA", "dnaN", "recA"]:
                if g in gene_names:
                    idx = gene_names.index(g)
                    X[i, idx] *= 4
    X = np.log1p(X)  # log transform (like scRNA)
    adata = AnnData(X, dtype=np.float32)
    adata.var_names = gene_names
    adata.obs['species'] = 'E. coli'
    adata.obs['cell_type'] = 'bacteria'
    return adata

def load_ecoli_real():
    """Placeholder for real E. coli scRNA‑seq dataset."""
    # In future, download from GEO using e.g., sc.datasets or custom loader
    raise NotImplementedError("Real E. coli dataset not yet implemented. Use synthetic for testing.")