# encoder/universal.py
"""Universal encoder from scRNA‑seq data to SUBIT 64‑state distribution.

This encoder uses functional gene sets (e.g., translation, energy, membrane)
that are conserved across species, allowing cross‑species comparisons.
"""

import numpy as np
import scanpy as sc
from scipy.special import softmax
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Improved default patterns for HUMAN (works with PBMC and many other tissues)
# These are actual gene symbols or short patterns.
# ----------------------------------------------------------------------

DEFAULT_WHO_PATTERNS = {
    "THEY": ["LAMP1", "CTSB", "CTSD", "GZMA", "PRF1", "dnaK", "groEL"],  # lysosomal / degradation / stress
    "YOU":  ["CD3D", "CD3E", "CD4", "CD8A", "CD19", "CD79A", "CD14", "FCGR3A"],  # lymphocyte markers / signalling
    "ME":   ["MT-", "COX", "NDUF", "ATP5", "SDHB", "UQCRC1", "CYCS"],   # mitochondrial / energy
    "WE":   ["RPL", "RPS", "EIF", "EEF", "FAU", "RACK1"],               # ribosomal / translation
}

DEFAULT_WHERE_PATTERNS = {
    "NORTH": ["RAN", "XPO1", "NUP93", "KPNA2", "KPNB1", "TNPO1"],         # nuclear transport / pores
    "WEST":  ["FN1", "COL1A1", "LAMA4", "TIMP1", "SERPINE1", "IL6", "TNF"], # extracellular matrix / secretion
    "EAST":  ["MKI67", "PCNA", "TOP2A", "NPM1", "HIST", "HMGB", "H2AFZ"],  # nucleus / chromatin / replication
    "SOUTH": ["CD99", "CD81", "BSG", "ITGB1", "ICAM1", "SELL", "CD44"],     # membrane / adhesion
}

DEFAULT_WHEN_PATTERNS = {
    "WINTER": ["CASP", "BAX", "BCL2L11", "GADD45", "CDKN1A", "HSPA1A"],   # apoptosis / stress / DNA damage
    "AUTUMN": ["CDK1", "CCNB1", "CCNB2", "AURKB", "PLK1", "TOP2A", "NUSAP1"], # G2/M / mitosis
    "SPRING": ["CCND1", "CCND2", "CDK4", "CDK6", "E2F1", "MYC", "MKI67"],   # G1 growth
    "SUMMER": ["PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "RRM1", "TYMS"], # S phase / replication
}


class UniversalEncoder:
    """
    Universal encoder from expression data to SUBIT 64‑state probabilities.
    
    Parameters
    ----------
    who_patterns : dict, optional
        Dictionary mapping WHO values ("THEY","YOU","ME","WE") to lists of gene substrings.
    where_patterns : dict, optional
        Mapping WHERE values ("NORTH","WEST","EAST","SOUTH") to lists of substrings.
    when_patterns : dict, optional
        Mapping WHEN values ("WINTER","AUTUMN","SPRING","SUMMER") to lists of substrings.
    normalize : bool, default True
        Whether to apply scanpy.pp.normalize_total (target_sum=1e4) and log1p to adata.
        If False, adata is assumed to be already normalized and log1p transformed.
    use_substring : bool, default False
        If True, match by substring (case-insensitive). If False, match exact gene symbols.
        For human data with exact symbols, set to False (faster, more precise).
    """
    
    def __init__(self,
                 who_patterns: Optional[Dict[str, List[str]]] = None,
                 where_patterns: Optional[Dict[str, List[str]]] = None,
                 when_patterns: Optional[Dict[str, List[str]]] = None,
                 normalize: bool = True,
                 use_substring: bool = False):
        self.who_patterns = who_patterns or DEFAULT_WHO_PATTERNS
        self.where_patterns = where_patterns or DEFAULT_WHERE_PATTERNS
        self.when_patterns = when_patterns or DEFAULT_WHEN_PATTERNS
        self.normalize = normalize
        self.use_substring = use_substring
        
        # Validate keys
        for d, name in zip([self.who_patterns, self.where_patterns, self.when_patterns],
                           ["WHO", "WHERE", "WHEN"]):
            expected = {"THEY","YOU","ME","WE"} if name == "WHO" else \
                       {"NORTH","WEST","EAST","SOUTH"} if name == "WHERE" else \
                       {"WINTER","AUTUMN","SPRING","SUMMER"}
            if set(d.keys()) != expected:
                raise ValueError(f"{name} patterns dict must contain exactly {expected}, got {set(d.keys())}")
    
    def _score_patterns(self, adata, patterns: Dict[str, List[str]]) -> np.ndarray:
        """
        Calculate mean expression for each pattern group.
        
        Returns
        -------
        np.ndarray, shape (n_cells, 4)
            Scores in the order of keys for the respective dimension.
        """
        # Determine key order (consistent with the dimension)
        if set(patterns.keys()) == {"THEY","YOU","ME","WE"}:
            key_order = ["THEY","YOU","ME","WE"]
        elif set(patterns.keys()) == {"NORTH","WEST","EAST","SOUTH"}:
            key_order = ["NORTH","WEST","EAST","SOUTH"]
        elif set(patterns.keys()) == {"WINTER","AUTUMN","SPRING","SUMMER"}:
            key_order = ["WINTER","AUTUMN","SPRING","SUMMER"]
        else:
            raise ValueError("Unrecognised pattern keys")
        
        scores = np.zeros((adata.n_obs, 4), dtype=np.float32)
        for i, key in enumerate(key_order):
            patterns_list = patterns[key]
            # Select matching genes
            if self.use_substring:
                # Substring matching (case-insensitive)
                selected_genes = []
                for gene in adata.var_names:
                    gene_lower = gene.lower()
                    if any(p.lower() in gene_lower for p in patterns_list):
                        selected_genes.append(gene)
            else:
                # Exact match (case‑sensitive, but we can also lower both sides)
                selected_genes = [g for g in adata.var_names if g in patterns_list or g.lower() in [p.lower() for p in patterns_list]]
            
            if len(selected_genes) == 0:
                logger.warning(f"No genes found for pattern {key} with patterns {patterns_list}. Using zero scores.")
                continue
            expr = adata[:, selected_genes].X
            if hasattr(expr, "mean"):
                mean_expr = np.asarray(expr.mean(axis=1)).ravel()
            else:
                mean_expr = np.mean(expr, axis=1)
            scores[:, i] = mean_expr
        return scores
    
    def _compute_who(self, adata) -> np.ndarray:
        scores = self._score_patterns(adata, self.who_patterns)
        # softmax along columns (axis=1)
        return softmax(scores, axis=1)
    
    def _compute_where(self, adata) -> np.ndarray:
        scores = self._score_patterns(adata, self.where_patterns)
        return softmax(scores, axis=1)
    
    def _compute_when(self, adata) -> np.ndarray:
        scores = self._score_patterns(adata, self.when_patterns)
        return softmax(scores, axis=1)
    
    def encode(self, adata) -> np.ndarray:
        """
        Compute P(SUBIT) for each cell in adata.
        
        Parameters
        ----------
        adata : AnnData
            Single‑cell expression data. If `self.normalize` is True, the object will be modified
            (normalised and log1p transformed) – use copy if needed.
        
        Returns
        -------
        P64 : np.ndarray, shape (n_cells, 64)
            Probability distribution over the 64 SUBIT states (indices 0..63).
        """
        # Optional preprocessing
        if self.normalize:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        
        P_who = self._compute_who(adata)       # (n_cells, 4)
        P_where = self._compute_where(adata)   # (n_cells, 4)
        P_when = self._compute_when(adata)     # (n_cells, 4)
        
        n_cells = adata.n_obs
        P64 = np.zeros((n_cells, 64), dtype=np.float32)
        for i in range(n_cells):
            # Kronecker product: who × where × when
            p64 = np.kron(P_who[i], np.kron(P_where[i], P_when[i]))
            P64[i] = p64
        return P64


# Convenience function for quick encoding
def encode_to_subit(adata, **kwargs) -> np.ndarray:
    """Shortcut to create a UniversalEncoder and call encode."""
    encoder = UniversalEncoder(**kwargs)
    return encoder.encode(adata)