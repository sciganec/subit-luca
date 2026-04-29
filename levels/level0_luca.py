# levels/level0_luca.py
"""Level 0: LUCA – Last Universal Common Ancestor.

Defines the minimal set of SUBIT states that could support life.
- WHO: only THEY (0) and WE (3)  [no YOU, no ME]
- WHERE: only EAST (2) and SOUTH (3)  [no NORTH, no WEST]
- WHEN: only SPRING (2), SUMMER (3), AUTUMN (1)  [no WINTER]

Total: 2 × 2 × 3 = 18 states.
"""

from typing import List
from levels.base import SubitLevel


class Level0_LUCA(SubitLevel):
    """LUCA level – 18 states corresponding to the first living automaton."""

    def allowed_states(self) -> List[int]:
        """Return a list of 18 indices representing LUCA's accessible states."""
        states = []
        # WHO: THEY (0) and WE (3)
        for who in (0, 3):
            # WHERE: EAST (2) and SOUTH (3)
            for where in (2, 3):
                # WHEN: SPRING (2), SUMMER (3), AUTUMN (1)
                for when in (2, 3, 1):
                    idx = (who << 4) | (where << 2) | when
                    states.append(idx)
        return states

    # Explicit list for quick reference or debugging
    EXPLICIT_STATES = [
        # THEY (0) + EAST (2) + SPRING (2) = 0b00001010 = 10
        0b00001010,
        # THEY + EAST + SUMMER (3) = 0b00001011 = 11
        0b00001011,
        # THEY + EAST + AUTUMN (1) = 0b00001001 = 9
        0b00001001,
        # THEY + SOUTH (3) + SPRING = 0b00001110 = 14
        0b00001110,
        # THEY + SOUTH + SUMMER = 0b00001111 = 15
        0b00001111,
        # THEY + SOUTH + AUTUMN = 0b00001101 = 13
        0b00001101,
        # WE (3) + EAST (2) + SPRING = 0b00111010 = 58
        0b00111010,
        # WE + EAST + SUMMER = 0b00111011 = 59
        0b00111011,
        # WE + EAST + AUTUMN = 0b00111001 = 57
        0b00111001,
        # WE + SOUTH (3) + SPRING = 0b00111110 = 62
        0b00111110,
        # WE + SOUTH + SUMMER = 0b00111111 = 63
        0b00111111,
        # WE + SOUTH + AUTUMN = 0b00111101 = 61
        0b00111101,
    ]

    # Optional: a method to verify that the number of states is correct
    @classmethod
    def count(cls) -> int:
        return len(cls().allowed_states())