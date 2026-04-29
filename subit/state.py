# subit/state.py
"""Core state definitions for SUBIT: WHO, WHERE, WHEN and their 6‑bit encoding."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


# ---------- Enums for dimensions ----------
# Values use the same bit patterns as the 2‑bit fields:
#   0b00 -> 0, 0b01 -> 1, 0b10 -> 2, 0b11 -> 3

class WHO:
    THEY = 0   # 0b00
    YOU  = 1   # 0b01
    ME   = 2   # 0b10
    WE   = 3   # 0b11

class WHERE:
    NORTH = 0  # 0b00
    WEST  = 1  # 0b01
    EAST  = 2  # 0b10
    SOUTH = 3  # 0b11

class WHEN:
    WINTER  = 0  # 0b00
    AUTUMN  = 1  # 0b01
    SPRING  = 2  # 0b10
    SUMMER  = 3  # 0b11


# Lookup tables for human‑readable names
WHO_NAMES = {WHO.THEY: "THEY", WHO.YOU: "YOU", WHO.ME: "ME", WHO.WE: "WE"}
WHERE_NAMES = {WHERE.NORTH: "NORTH", WHERE.WEST: "WEST", WHERE.EAST: "EAST", WHERE.SOUTH: "SOUTH"}
WHEN_NAMES = {WHEN.WINTER: "WINTER", WHEN.AUTUMN: "AUTUMN", WHEN.SPRING: "SPRING", WHEN.SUMMER: "SUMMER"}

# Reverse maps (string -> code)
WHO_FROM_NAME = {v: k for k, v in WHO_NAMES.items()}
WHERE_FROM_NAME = {v: k for k, v in WHERE_NAMES.items()}
WHEN_FROM_NAME = {v: k for k, v in WHEN_NAMES.items()}


# ---------- SUBIT State class ----------
@dataclass(frozen=True, order=True)
class SubitState:
    """A concrete state in the 64‑dimensional SUBIT space."""
    who: int   # 0..3
    where: int # 0..3
    when: int  # 0..3

    def __post_init__(self):
        """Validate that all components are in range 0..3."""
        if not (0 <= self.who <= 3):
            raise ValueError(f"who must be 0..3, got {self.who}")
        if not (0 <= self.where <= 3):
            raise ValueError(f"where must be 0..3, got {self.where}")
        if not (0 <= self.when <= 3):
            raise ValueError(f"when must be 0..3, got {self.when}")

    def to_index(self) -> int:
        """
        Encode state into a 6‑bit integer (0..63).

        Bits: [who (2 bits)] [where (2 bits)] [when (2 bits)]
        """
        return (self.who << 4) | (self.where << 2) | self.when

    @classmethod
    def from_index(cls, idx: int) -> SubitState:
        """Decode a 6‑bit integer back into a SubitState."""
        if not (0 <= idx <= 63):
            raise ValueError(f"index must be 0..63, got {idx}")
        who = (idx >> 4) & 0b11
        where = (idx >> 2) & 0b11
        when = idx & 0b11
        return cls(who=who, where=where, when=when)

    @classmethod
    def from_names(cls, who_name: str, where_name: str, when_name: str) -> SubitState:
        """Create state from human‑readable names, e.g. 'WE', 'EAST', 'SUMMER'."""
        who = WHO_FROM_NAME[who_name.upper()]
        where = WHERE_FROM_NAME[where_name.upper()]
        when = WHEN_FROM_NAME[when_name.upper()]
        return cls(who=who, where=where, when=when)

    def to_names(self) -> Tuple[str, str, str]:
        """Return (who_name, where_name, when_name)."""
        return (WHO_NAMES[self.who], WHERE_NAMES[self.where], WHEN_NAMES[self.when])

    def __str__(self) -> str:
        who_name, where_name, when_name = self.to_names()
        return f"{who_name}/{where_name}/{when_name}"

    def __repr__(self) -> str:
        return f"SubitState(who={self.who}, where={self.where}, when={self.when})"


# ---------- Utility functions ----------
def all_states() -> List[SubitState]:
    """Return a list of all 64 possible SUBIT states in index order (0..63)."""
    return [SubitState.from_index(i) for i in range(64)]


def index_to_triplet(idx: int) -> Tuple[int, int, int]:
    """Convert index to (who, where, when) integers."""
    return ((idx >> 4) & 0b11, (idx >> 2) & 0b11, idx & 0b11)


def triplet_to_index(who: int, where: int, when: int) -> int:
    """Convert (who, where, when) integers to index."""
    return (who << 4) | (where << 2) | when


# ---------- Pre‑computed arrays for fast lookups ----------
# Names array: at index i -> string like "WE/EAST/SUMMER"
STATE_NAMES = [str(SubitState.from_index(i)) for i in range(64)]

# Arrays of components for vectorised operations (useful for metrics)
WHO_OF_INDEX = [(i >> 4) & 0b11 for i in range(64)]
WHERE_OF_INDEX = [(i >> 2) & 0b11 for i in range(64)]
WHEN_OF_INDEX = [i & 0b11 for i in range(64)]