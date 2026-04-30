# metrics/luca.py
"""LUCA‑related metrics for SUBIT probability distributions.

LUCA (Last Universal Common Ancestor) corresponds to state index 0 (`000000`)
with zero '1' bits. The LUCA distance of a distribution is the expected number
of '1' bits (population count) under that distribution. This measures how
“far” the cell’s state is from the minimal ancestral state.
"""

import numpy as np
from typing import Union


# Pre‑compute popcount (number of '1' bits) for all 64 states
_POPCOUNT = np.array([bin(i).count('1') for i in range(64)], dtype=np.float32)


def luca_distance(p: np.ndarray) -> Union[float, np.ndarray]:
    """
    Compute the expected number of '1' bits (population count) for a SUBIT distribution.
    
    Parameters
    ----------
    p : np.ndarray
        Probability distribution(s). Shape (64,) for a single cell, or (n_cells, 64)
        for many cells.
    
    Returns
    -------
    float or np.ndarray
        Expected popcount (LUCA distance). For 1‑D input returns a scalar,
        for 2‑D returns an array of shape (n_cells,).
    """
    p = np.asarray(p)
    # If 1‑D, add a batch dimension temporarily
    if p.ndim == 1:
        p = p.reshape(1, -1)
        dist = np.sum(p * _POPCOUNT, axis=1)[0]
        return float(dist)
    else:
        return np.sum(p * _POPCOUNT, axis=1)


def luca_projection(p: np.ndarray, level0_states) -> Union[float, np.ndarray]:
    """
    Compute the probability mass that lies on LUCA‑allowed states (level 0 subset).
    
    Parameters
    ----------
    p : np.ndarray
        Probability distribution(s). Shape (64,) or (n_cells, 64).
    level0_states : list or array of ints
        List of allowed LUCA state indices (e.g., from Level0_LUCA().allowed_states()).
    
    Returns
    -------
    float or np.ndarray
        Total probability on allowed LUCA states. For 1‑D input returns scalar,
        for 2‑D returns array of shape (n_cells,).
    """
    p = np.asarray(p)
    allowed_mask = np.zeros(64, dtype=bool)
    allowed_mask[level0_states] = True
    if p.ndim == 1:
        return np.sum(p[allowed_mask])
    else:
        return np.sum(p[:, allowed_mask], axis=1)