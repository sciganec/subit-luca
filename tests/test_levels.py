# tests/test_levels.py
"""Unit tests for evolutionary levels (levels/base.py and levels/level0_luca.py)."""

import pytest
import numpy as np
from levels.base import SubitLevel
from levels.level0_luca import Level0_LUCA
from subit.state import WHO, WHERE, WHEN, index_to_triplet


# ---------- Test Level0_LUCA ----------
def test_luca_allowed_states_count():
    """LUCA must have exactly 18 states."""
    luca = Level0_LUCA()
    states = luca.allowed_states()
    assert len(states) == 18
    # Check that there are no duplicates
    assert len(set(states)) == 18

def test_luca_allowed_states_within_range():
    """All LUCA states must be valid indices 0..63."""
    luca = Level0_LUCA()
    for idx in luca.allowed_states():
        assert 0 <= idx <= 63

def test_luca_state_constraints():
    """
    Verify that each LUCA state conforms to:
    - WHO ∈ {THEY (0), WE (3)}
    - WHERE ∈ {EAST (2), SOUTH (3)}
    - WHEN ∈ {SPRING (2), SUMMER (3), AUTUMN (1)}
    """
    luca = Level0_LUCA()
    for idx in luca.allowed_states():
        who, where, when = index_to_triplet(idx)
        assert who in (WHO.THEY, WHO.WE)
        assert where in (WHERE.EAST, WHERE.SOUTH)
        assert when in (WHEN.SPRING, WHEN.SUMMER, WHEN.AUTUMN)

def test_luca_no_extra_states():
    """
    Ensure that no state outside the allowed set is included.
    We generate all 64 states, filter by constraints, and compare.
    """
    expected = set()
    for who in (WHO.THEY, WHO.WE):
        for where in (WHERE.EAST, WHERE.SOUTH):
            for when in (WHEN.SPRING, WHEN.SUMMER, WHEN.AUTUMN):
                idx = (who << 4) | (where << 2) | when
                expected.add(idx)
    luca = Level0_LUCA()
    assert set(luca.allowed_states()) == expected

def test_luca_explicit_states_match():
    """Check that the hardcoded EXPLICIT_STATES list equals the generated states."""
    luca = Level0_LUCA()
    generated = set(luca.allowed_states())
    explicit = set(luca.EXPLICIT_STATES)
    assert generated == explicit


# ---------- Test SubitLevel.project (using LUCA as concrete example) ----------
def test_project_1d():
    """Project a 1D probability array (64,) onto LUCA subspace."""
    luca = Level0_LUCA()
    # Create a uniform distribution over all 64 states
    p_full = np.ones(64) / 64.0
    p_proj = luca.project(p_full)
    # Forbidden states should be zero
    for idx in range(64):
        if idx not in luca.allowed_states():
            assert p_proj[idx] == 0.0
    # Allowed states should sum to 1
    assert np.isclose(p_proj.sum(), 1.0)
    # Allowed states should have non‑zero probability (equal within allowed set)
    # Since input was uniform, after zeroing out forbidden states we renormalize
    expected_prob = 1.0 / len(luca.allowed_states())
    for idx in luca.allowed_states():
        assert np.isclose(p_proj[idx], expected_prob)

def test_project_1d_zero_on_allowed():
    """If input has zero probability on all allowed states, project should fall back to uniform over allowed."""
    luca = Level0_LUCA()
    p = np.zeros(64)
    # Give probability only to a forbidden state
    p[0] = 1.0   # 0 is THEY/NORTH/WINTER – forbidden for LUCA (NORTH not allowed)
    p_proj = luca.project(p)
    # After projection, probability should be uniform over allowed states
    assert np.isclose(p_proj.sum(), 1.0)
    expected = 1.0 / len(luca.allowed_states())
    for idx in luca.allowed_states():
        assert np.isclose(p_proj[idx], expected)
    for idx in range(64):
        if idx not in luca.allowed_states():
            assert p_proj[idx] == 0.0

def test_project_2d():
    """Project a 2D array (n_cells, 64)."""
    luca = Level0_LUCA()
    n_cells = 5
    p_full = np.ones((n_cells, 64)) / 64.0
    p_proj = luca.project(p_full)
    assert p_proj.shape == (n_cells, 64)
    # Check each row
    for i in range(n_cells):
        row = p_proj[i]
        # Forbidden states zero
        for idx in range(64):
            if idx not in luca.allowed_states():
                assert row[idx] == 0.0
        # Sum to 1
        assert np.isclose(row.sum(), 1.0)
        # Allowed states uniform
        expected = 1.0 / len(luca.allowed_states())
        for idx in luca.allowed_states():
            assert np.isclose(row[idx], expected)

def test_project_2d_zero_rows():
    """If some rows have zero probability on allowed states, they fall back to uniform."""
    luca = Level0_LUCA()
    n_cells = 3
    p = np.zeros((n_cells, 64))
    # Row 0: probability on allowed state
    allowed = luca.allowed_states()[0]
    p[0, allowed] = 1.0
    # Row 1: probability on a forbidden state
    p[1, 0] = 1.0   # 0 is forbidden
    # Row 2: all zeros
    p[2, :] = 0.0
    p_proj = luca.project(p)
    # Row 0 should keep its probability on that allowed state
    assert np.isclose(p_proj[0, allowed], 1.0)
    # Row 1 and 2 should become uniform over allowed states
    expected_uniform = 1.0 / len(luca.allowed_states())
    for idx in luca.allowed_states():
        assert np.isclose(p_proj[1, idx], expected_uniform)
        assert np.isclose(p_proj[2, idx], expected_uniform)
    # Forbidden states zero in all rows
    for i in range(n_cells):
        for idx in range(64):
            if idx not in luca.allowed_states():
                assert p_proj[i, idx] == 0.0


# ---------- Test inheritance and abstract method ----------
def test_subit_level_abstract():
    """SubitLevel cannot be instantiated directly (abstract)."""
    with pytest.raises(TypeError):
        SubitLevel()  # abstract, should raise

class DummyLevel(SubitLevel):
    def allowed_states(self):
        return [0, 1, 2]

def test_concrete_level():
    """A concrete subclass must implement allowed_states."""
    level = DummyLevel()
    assert level.allowed_states() == [0, 1, 2]
    # Test project with a concrete simple level
    p = np.ones(64) / 64.0
    p_proj = level.project(p)
    # Only states 0,1,2 allowed, others zero
    for idx in range(64):
        if idx in (0,1,2):
            assert p_proj[idx] > 0
        else:
            assert p_proj[idx] == 0.0
    assert np.isclose(p_proj.sum(), 1.0)