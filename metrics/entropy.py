# metrics/entropy.py
"""Entropy metrics for SUBIT probability distributions.

Shannon entropy measures the uncertainty or "plasticity" of a cell's state distribution.
Higher entropy indicates a more mixed, less committed state; lower entropy suggests
a well‑defined, dominant state (e.g., highly specialised cell or one stuck in a single
archetype).
"""

import numpy as np
from typing import Union


def entropy(p: np.ndarray, base: float = np.e) -> Union[float, np.ndarray]:
    """
    Compute Shannon entropy for a probability distribution over 64 SUBIT states.
    
    Parameters
    ----------
    p : np.ndarray
        Probability distribution(s). Can be 1‑D (shape (64,)) for a single cell,
        or 2‑D (shape (n_cells, 64)) for many cells.
    base : float, default np.e
        Logarithm base. Use `np.e` for nats, `2.0` for bits, `10.0` for dits.
    
    Returns
    -------
    float or np.ndarray
        Entropy value(s). For 1‑D input returns a scalar; for 2‑D returns an array
        of shape (n_cells,).
    
    Notes
    -----
    - Input probabilities are assumed to sum to 1 along the last axis.
    - Zero probabilities are handled gracefully (0 * log(0) → 0).
    """
    p = np.asarray(p)
    # Avoid taking log of zero
    p_safe = np.where(p > 0, p, 1.0)
    log_p = np.log(p_safe) / np.log(base)
    entropy_vals = -np.sum(p * log_p, axis=-1)
    # Remove possible small negatives due to floating point
    entropy_vals = np.maximum(entropy_vals, 0.0)
    if entropy_vals.ndim == 0:
        return float(entropy_vals)
    return entropy_vals


def normalized_entropy(p: np.ndarray, base: float = np.e) -> Union[float, np.ndarray]:
    """
    Normalised entropy: entropy divided by maximum possible entropy (log(64)/log(base)).
    Returns values in [0, 1] where 1 means uniform distribution over all 64 states.
    """
    max_entropy = np.log(64) / np.log(base)
    e = entropy(p, base=base)
    return e / max_entropy


def relative_entropy(p: np.ndarray, q: np.ndarray, base: float = np.e) -> np.ndarray:
    """
    Kullback‑Leibler divergence D_KL(p || q) for SUBIT distributions.
    
    Parameters
    ----------
    p : np.ndarray
        Observed distribution(s), shape (64,) or (n_cells, 64).
    q : np.ndarray
        Reference distribution(s), same shape as p.
    base : float, default np.e
        Logarithm base.
    
    Returns
    -------
    np.ndarray
        KL divergence, scalar or array of shape (n_cells,).
    """
    p = np.asarray(p)
    q = np.asarray(q)
    # Avoid division by zero or log(0)
    q_safe = np.where(q > 0, q, 1.0)
    p_safe = np.where(p > 0, p, 1.0)
    kl = np.sum(p * (np.log(p_safe) - np.log(q_safe)), axis=-1) / np.log(base)
    kl = np.maximum(kl, 0.0)  # numerical safety
    return kl