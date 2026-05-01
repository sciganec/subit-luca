# subit/encoding.py
"""Low‑level encoding utilities for SUBIT probability distributions.

This module provides functions to convert between factor distributions
(WHO, WHERE, WHEN) and the joint 64‑state distribution, as well as
marginalisation and subspace projection.
"""

import numpy as np
from typing import Tuple


def kron3(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> np.ndarray:
    """
    Compute the Kronecker product of three probability vectors for each row.

    For each row i, out[i] = kron(pa[i], kron(pb[i], pc[i])).
    The resulting vector has length 4*4*4 = 64.

    Parameters
    ----------
    pa : np.ndarray, shape (n, 4)
        Probabilities for WHO (order: THEY, YOU, ME, WE)
    pb : np.ndarray, shape (n, 4)
        Probabilities for WHERE (order: NORTH, WEST, EAST, SOUTH)
    pc : np.ndarray, shape (n, 4)
        Probabilities for WHEN (order: WINTER, AUTUMN, SPRING, SUMMER)

    Returns
    -------
    np.ndarray, shape (n, 64)
        Joint distribution over SUBIT states (index 0..63).
    """
    n = pa.shape[0]
    out = np.zeros((n, 64), dtype=pa.dtype)
    for i in range(n):
        out[i] = np.kron(pa[i], np.kron(pb[i], pc[i]))
    return out


def kron3_fast(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> np.ndarray:
    """
    Faster (vectorised) Kronecker product for three factor distributions.

    This uses broadcasting and reshaping instead of a loop.
    Assumes input arrays of shape (n, 4). The resulting array has shape (n, 64).

    Parameters
    ----------
    pa, pb, pc : np.ndarray, shape (n, 4)

    Returns
    -------
    np.ndarray, shape (n, 64)
    """
    # Outer product of pb and pc: (n, 4, 4)
    pbc = pb[..., :, np.newaxis] * pc[..., np.newaxis, :]
    # Reshape to (n, 16)
    pbc = pbc.reshape(-1, 16)
    # Outer product of pa and pbc: (n, 4, 16)
    p = pa[..., :, np.newaxis] * pbc[..., np.newaxis, :]
    # Reshape to (n, 64)
    return p.reshape(-1, 64)


def marginals(p64: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute marginal distributions over WHO, WHERE, WHEN from joint 64‑state distribution.

    Parameters
    ----------
    p64 : np.ndarray, shape (n, 64)
        Joint probabilities over states (index 0..63).

    Returns
    -------
    tuple of np.ndarray: (p_who, p_where, p_when), each shape (n, 4)
        Marginals in the same order as used in the Kronecker product.
        p_who: [THEY, YOU, ME, WE]
        p_where: [NORTH, WEST, EAST, SOUTH]
        p_when: [WINTER, AUTUMN, SPRING, SUMMER]
    """
    n = p64.shape[0]
    p_who = np.zeros((n, 4), dtype=p64.dtype)
    p_where = np.zeros((n, 4), dtype=p64.dtype)
    p_when = np.zeros((n, 4), dtype=p64.dtype)

    for i in range(n):
        # Reshape to (4,4,4) – who, where, when
        cube = p64[i].reshape(4, 4, 4)
        # Sum over where and when for each who
        p_who[i] = cube.sum(axis=(1, 2))  # shape (4,)
        # Sum over who and when for each where
        p_where[i] = cube.sum(axis=(0, 2))  # shape (4,)
        # Sum over who and where for each when
        p_when[i] = cube.sum(axis=(0, 1))   # shape (4,)
    return p_who, p_where, p_when


def project_to_subspace(p64: np.ndarray, allowed_states: np.ndarray) -> np.ndarray:
    """
    Project a joint distribution onto a subspace by zeroing out forbidden states
    and renormalising.

    Parameters
    ----------
    p64 : np.ndarray, shape (n, 64)
        Joint distribution(s).
    allowed_states : np.ndarray, shape (k,)
        List of state indices (0..63) that are allowed.

    Returns
    -------
    np.ndarray, shape (n, 64)
        Projected distribution(s).
    """
    p_out = p64.copy()
    forbidden_mask = np.ones(64, dtype=bool)
    forbidden_mask[allowed_states] = False
    p_out[:, forbidden_mask] = 0.0
    row_sum = p_out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return p_out / row_sum


def argmax_state(p64: np.ndarray) -> np.ndarray:
    """
    Return the index (0..63) of the most probable SUBIT state for each cell.

    Parameters
    ----------
    p64 : np.ndarray, shape (n, 64)

    Returns
    -------
    np.ndarray, shape (n,)
        State indices.
    """
    return np.argmax(p64, axis=1)