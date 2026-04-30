# subit/space.py
"""Define the 64‑state SUBIT space and operations on it."""

from typing import List, Set, Tuple
import numpy as np
from subit.state import index_to_triplet, triplet_to_index, WHO_OF_INDEX, WHERE_OF_INDEX, WHEN_OF_INDEX


class SubitSpace:
    """
    A 64‑state space where each state is a 6‑bit integer (0..63).
    
    The space is a 6‑dimensional hypercube. Neighbors are states differing by one bit
    (Hamming distance = 1). This class provides static methods for navigation
    and distance calculations.
    """
    
    # Pre‑computed neighbor list for every index (0..63)
    _neighbors_cache: List[List[int]] = []
    _hamming_cache: np.ndarray = None
    
    @classmethod
    def _build_cache(cls):
        """Build neighbour and Hamming distance caches for all 64 states."""
        if cls._neighbors_cache:
            return
        cls._neighbors_cache = [[] for _ in range(64)]
        for i in range(64):
            # Flip each of the 6 bits
            for bit in range(6):
                j = i ^ (1 << bit)
                cls._neighbors_cache[i].append(j)
        # Hamming distance matrix (64 x 64) – optional, but useful for metrics
        cls._hamming_cache = np.zeros((64, 64), dtype=np.uint8)
        for i in range(64):
            for j in range(64):
                cls._hamming_cache[i, j] = cls.hamming(i, j)
    
    @staticmethod
    def hamming(a: int, b: int) -> int:
        """Hamming distance between two 6‑bit indices (number of differing bits)."""
        # XOR and count bits (popcount)
        return (a ^ b).bit_count()
    
    @classmethod
    def neighbors(cls, state: int) -> List[int]:
        """
        Return a list of states that differ by exactly one bit (Hamming distance 1).
        
        Parameters
        ----------
        state : int
            Index in 0..63.
        
        Returns
        -------
        list[int]
            List of up to 6 neighbour indices.
        """
        if not (0 <= state < 64):
            raise ValueError(f"state must be in 0..63, got {state}")
        cls._build_cache()
        return cls._neighbors_cache[state][:]
    
    @classmethod
    def distance_matrix(cls) -> np.ndarray:
        """Return a 64x64 matrix of pairwise Hamming distances."""
        cls._build_cache()
        return cls._hamming_cache.copy()
    
    @staticmethod
    def components(state: int) -> Tuple[int, int, int]:
        """Return (who, where, when) components for a given state index."""
        return index_to_triplet(state)
    
    @staticmethod
    def from_components(who: int, where: int, when: int) -> int:
        """Return state index from (who, where, when)."""
        return triplet_to_index(who, where, when)
    
    @classmethod
    def who_of(cls, state: int) -> int:
        """Return the WHO component (0..3) of the state."""
        return WHO_OF_INDEX[state]
    
    @classmethod
    def where_of(cls, state: int) -> int:
        """Return the WHERE component (0..3) of the state."""
        return WHERE_OF_INDEX[state]
    
    @classmethod
    def when_of(cls, state: int) -> int:
        """Return the WHEN component (0..3) of the state."""
        return WHEN_OF_INDEX[state]
    
    @classmethod
    def set_who(cls, state: int, new_who: int) -> int:
        """Replace WHO component, keep WHERE and WHEN."""
        where = cls.where_of(state)
        when = cls.when_of(state)
        return cls.from_components(new_who, where, when)
    
    @classmethod
    def set_where(cls, state: int, new_where: int) -> int:
        """Replace WHERE component, keep WHO and WHEN."""
        who = cls.who_of(state)
        when = cls.when_of(state)
        return cls.from_components(who, new_where, when)
    
    @classmethod
    def set_when(cls, state: int, new_when: int) -> int:
        """Replace WHEN component, keep WHO and WHERE."""
        who = cls.who_of(state)
        where = cls.where_of(state)
        return cls.from_components(who, where, new_when)
    
    @classmethod
    def all_states(cls) -> List[int]:
        """Return list of all 64 indices (0..63)."""
        return list(range(64))
    
    @classmethod
    def sublattice(cls, who_set: Set[int], where_set: Set[int], when_set: Set[int]) -> List[int]:
        """
        Return states where WHO ∈ who_set, WHERE ∈ where_set, WHEN ∈ when_set.
        
        Parameters
        ----------
        who_set : set of int (0..3)
        where_set : set of int (0..3)
        when_set : set of int (0..3)
        
        Returns
        -------
        list[int]
            Indices of all states satisfying the constraints.
        """
        states = []
        for who in who_set:
            for where in where_set:
                for when in when_set:
                    states.append(triplet_to_index(who, where, when))
        return states


# Convenience functions for quick access (without class)
def hamming(a: int, b: int) -> int:
    """Hamming distance between two 6‑bit indices."""
    return SubitSpace.hamming(a, b)

def neighbors(state: int) -> List[int]:
    """Neighbors of a state (Hamming distance 1)."""
    return SubitSpace.neighbors(state)