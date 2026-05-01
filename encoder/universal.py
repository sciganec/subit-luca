# encoder/universal.py
"""Universal encoder from scRNA‑seq data to SUBIT 64‑state distribution.

This encoder uses functional gene sets (e.g., translation, energy, membrane)
with patterns that work across species – human, E. coli, yeast, etc.
"""

import numpy as np
import scanpy as sc
from scipy.special import softmax
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Universal functional patterns (works for human, bacteria, archaea)
# Each entry contains either:
#   - exact human gene symbols (capitalised)
#   - short substring motifs (e.g., "rpl", "sec", "atp") for broad matching
# Patterns are case‑insensitive when `use_substring=True`.
# ----------------------------------------------------------------------

DEFAULT_WHO_PATTERNS = {
    "THEY": [
        # lysosomal / degradation / stress (human)
        "LAMP1", "CTSB", "CTSD", "GZMA", "PRF1",
        # bacterial/universal stress & proteolysis
        "lon", "clpP", "htpG", "groEL", "dnaK", "clpB", "hslV"
    ],
    "YOU": [
        # immune / signalling (human)
        "CD3D", "CD3E", "CD4", "CD8A", "CD19", "CD79A", "CD14", "FCGR3A", "CD163",
        # bacterial chemotaxis / two‑component systems
        "che", "tar", "tsr", "aer", "phoB", "ompR"
    ],
    "ME": [
        # mitochondrial / energy (human & universal)
        "MT-CO1", "MT-ND1", "ATP5", "NDUF", "COX", "SDHB", "UQCRC1", "CYCS",
        # bacterial ATP synthase / respiratory chain
        "atpA", "atpD", "atpG", "ndh", "nuo", "cydA", "cyo"
    ],
    "WE": [
        # ribosomal / translation (universal)
        "RPL", "RPS", "EIF", "EEF", "FAU", "RACK1",
        # bacterial ribosomal proteins
        "rpl", "rps", "rpm", "rbfA", "infA", "tuf"
    ],
}

DEFAULT_WHERE_PATTERNS = {
    "NORTH": [
        # nuclear transport / pores (human)
        "RAN", "XPO1", "NUP93", "KPNA2", "KPNB1", "TNPO1",
        # bacterial secretion / transport systems (e.g., Sec, Tat, Type III)
        "secA", "secY", "secE", "tatA", "tatC", "tolC", "ompR"
    ],
    "WEST": [
        # extracellular / secreted (human)
        "FN1", "COL1A1", "LAMA4", "TIMP1", "SERPINE1", "IL6", "TNF",
        # bacterial exoproteins / surface adhesins
        "flagellin", "fimA", "csgA", "hlyA", "lpp", "ompA", "ompC"
    ],
    "EAST": [
        # nucleus / chromatin / replication (human)
        "MKI67", "PCNA", "TOP2A", "NPM1", "HIST", "HMGB", "H2AFZ",
        # bacterial nucleoid / DNA replication
        "dnaA", "dnaB", "dnaN", "gyrA", "gyrB", "recA", "ssb", "hup", "ihf"
    ],
    "SOUTH": [
        # membrane / ER / surface (human)
        "CD99", "CD81", "BSG", "ITGB1", "ICAM1", "SELL", "CD44",
        # bacterial inner/outer membrane proteins
        "secY", "secE", "ompA", "ompC", "ompF", "lptD", "bamA", "ftsZ"
    ],
}

DEFAULT_WHEN_PATTERNS = {
    "WINTER": [
        # apoptosis / stress / DNA damage (human)
        "CASP", "BAX", "BCL2L11", "GADD45", "CDKN1A", "HSPA1A",
        # bacterial stress response / stringent response
        "relA", "spoT", "rpoS", "dnaK", "groEL", "lon", "clpP"
    ],
    "AUTUMN": [
        # G2/M / mitosis (human)
        "CDK1", "CCNB1", "CCNB2", "AURKB", "PLK1", "TOP2A", "NUSAP1",
        # bacterial cell division
        "ftsZ", "ftsA", "ftsQ", "ftsK", "divIVA", "minC", "minD"
    ],
    "SPRING": [
        # G1 growth (human)
        "CCND1", "CCND2", "CDK4", "CDK6", "E2F1", "MYC", "MKI67",
        # bacterial growth (ribosome biogenesis, metabolic activation)
        "rpoB", "rpoC", "rpl", "rps", "tuf", "fusA", "glyA"
    ],
    "SUMMER": [
        # S phase / replication (human)
        "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "RRM1", "TYMS",
        # bacterial DNA replication
        "dnaA", "dnaB", "dnaC", "dnaG", "dnaN", "gyrA", "gyrB", "topA"
    ],
}


class UniversalEncoder:
    """
    Universal encoder from expression data to SUBIT 64‑state probabilities.
    
    Parameters
    ----------
    who_patterns : dict, optional
        Dictionary mapping WHO values ("THEY","YOU","ME","WE") to lists of gene patterns.
    where_patterns : dict, optional
        Mapping WHERE values ("NORTH","WEST","EAST","SOUTH") to lists of patterns.
    when_patterns : dict, optional
        Mapping WHEN values ("WINTER","AUTUMN","SPRING","SUMMER") to lists of patterns.
    normalize : bool, default True
        Whether to apply scanpy.pp.normalize_total (target_sum=1e4) and log1p to adata.
        If False, adata is assumed to be already normalized and log1p transformed.
    use_substring : bool, default True
        If True, match patterns as substrings (case‑insensitive). Useful for cross‑species.
        If False, match exact gene symbols (faster for human data).
    """
    
    def __init__(self,
                 who_patterns: Optional[Dict[str, List[str]]] = None,
                 where_patterns: Optional[Dict[str, List[str]]] = None,
                 when_patterns: Optional[Dict[str, List[str]]] = None,
                 normalize: bool = True,
                 use_substring: bool = True):
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
                # Substring matching (case‑insensitive)
                selected_genes = []
                for gene in adata.var_names:
                    gene_lower = gene.lower()
                    if any(p.lower() in gene_lower for p in patterns_list):
                        selected_genes.append(gene)
            else:
                # Exact match (case‑sensitive, but we also try lower if needed)
                selected_genes = [g for g in adata.var_names 
                                  if g in patterns_list or g.lower() in [p.lower() for p in patterns_list]]
            
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