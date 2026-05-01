# subit/__init__.py
"""SUBIT core: state definition and 64‑state space."""

from .state import SubitState, all_states, index_to_triplet, triplet_to_index
from .space import SubitSpace, hamming, neighbors

__all__ = [
    "SubitState",
    "all_states",
    "index_to_triplet",
    "triplet_to_index",
    "SubitSpace",
    "hamming",
    "neighbors",
]