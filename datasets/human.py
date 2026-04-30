# datasets/human.py
"""Load human single‑cell RNA‑seq datasets and prepare them for SUBIT encoding."""

import scanpy as sc
import numpy as np
from anndata import AnnData
from typing import Optional


def load_human_pbmc(normalize: bool = False) -> AnnData:
    """
    Load the built‑in PBMC 3k dataset (10x Genomics).
    
    The data is filtered to remove cells with too few genes and genes expressed
    in too few cells. No normalisation or log transformation is applied by default
    (set `normalize=True` to run standard scanpy normalisation and log1p).
    
    Parameters
    ----------
    normalize : bool, default False
        If True, apply `sc.pp.normalize_total(target_sum=1e4)` and `sc.pp.log1p`.
    
    Returns
    -------
    AnnData
        Raw (or normalized) expression matrix with `.var_names` as gene symbols
        and `.obs` containing basic QC metrics.
    """
    adata = sc.datasets.pbmc3k()
    # Basic filtering to remove low‑quality cells/genes (common practice)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    return adata


def load_human_pbmc_with_cycle_scores() -> AnnData:
    """
    Load PBMC 3k data and add cell cycle phase scores (S and G2M).
    
    This is useful for validating the WHEN dimension of SUBIT.
    
    Returns
    -------
    AnnData
        Same as `load_human_pbmc()` but with additional `.obs` columns:
        'S_score', 'G2M_score', 'phase' (G1/S/G2M).
    """
    adata = load_human_pbmc(normalize=True)
    # Known cell cycle genes for human (from scanpy's example)
    s_genes = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6']
    g2m_genes = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'KIF20B']
    # Keep only genes present in the dataset
    s_genes = [g for g in s_genes if g in adata.var_names]
    g2m_genes = [g for g in g2m_genes if g in adata.var_names]
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    return adata


def load_human_pbmc_subset(n_cells: int = 500) -> AnnData:
    """
    Load a random subset of PBMC cells (for quick experimentation).
    
    Parameters
    ----------
    n_cells : int, default 500
        Number of cells to randomly subsample.
    
    Returns
    -------
    AnnData
        Subsampled AnnData object.
    """
    adata = load_human_pbmc(normalize=False)
    if n_cells < adata.n_obs:
        np.random.seed(42)   # for reproducibility
        idx = np.random.choice(adata.n_obs, n_cells, replace=False)
        adata = adata[idx].copy()
    return adata


# Additional human datasets can be added here, for example:
#
# def load_human_pancreas():
#     """Load human pancreas dataset (Lawlor et al.)."""
#     adata = sc.datasets.pancreas()
#     ... return preprocessed data