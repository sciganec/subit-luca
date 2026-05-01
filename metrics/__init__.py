# metrics/__init__.py
"""Metrics for SUBIT distributions: entropy, LUCA distance, complexity."""

from .entropy import entropy, normalized_entropy, relative_entropy
from .luca import luca_distance, luca_projection

def complexity(p):
    """Combined measure: entropy + LUCA distance."""
    return entropy(p) + luca_distance(p)

__all__ = [
    "entropy",
    "normalized_entropy",
    "relative_entropy",
    "luca_distance",
    "luca_projection",
    "complexity",
]