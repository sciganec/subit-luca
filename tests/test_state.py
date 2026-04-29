# tests/test_state.py
"""Unit tests for subit/state.py."""

import pytest
from subit.state import (
    SubitState,
    WHO,
    WHERE,
    WHEN,
    WHO_NAMES,
    WHERE_NAMES,
    WHEN_NAMES,
    WHO_FROM_NAME,
    WHERE_FROM_NAME,
    WHEN_FROM_NAME,
    all_states,
    index_to_triplet,
    triplet_to_index,
    STATE_NAMES,
    WHO_OF_INDEX,
    WHERE_OF_INDEX,
    WHEN_OF_INDEX,
)


# ---------- Test SubitState ----------
def test_subit_state_creation():
    s = SubitState(who=WHO.ME, where=WHERE.EAST, when=WHEN.SPRING)
    assert s.who == WHO.ME
    assert s.where == WHERE.EAST
    assert s.when == WHEN.SPRING

def test_subit_state_invalid_who():
    with pytest.raises(ValueError, match="who must be 0..3"):
        SubitState(who=4, where=WHERE.EAST, when=WHEN.SPRING)

def test_subit_state_invalid_where():
    with pytest.raises(ValueError, match="where must be 0..3"):
        SubitState(who=WHO.ME, where=5, when=WHEN.SPRING)

def test_subit_state_invalid_when():
    with pytest.raises(ValueError, match="when must be 0..3"):
        SubitState(who=WHO.ME, where=WHERE.EAST, when=99)


# ---------- Test to_index / from_index ----------
@pytest.mark.parametrize("who,where,when,expected_index", [
    (WHO.THEY, WHERE.NORTH, WHEN.WINTER, 0b00000000),  # 0
    (WHO.YOU,  WHERE.WEST,  WHEN.AUTUMN, 0b01010101),  # 0b01 01 01 = 21
    (WHO.ME,   WHERE.EAST,  WHEN.SPRING, 0b10101010),  # 0b10 10 10 = 42
    (WHO.WE,   WHERE.SOUTH, WHEN.SUMMER, 0b11111111),  # 63
])
def test_to_index(who, where, when, expected_index):
    s = SubitState(who=who, where=where, when=when)
    assert s.to_index() == expected_index

@pytest.mark.parametrize("idx,expected_who,expected_where,expected_when", [
    (0, WHO.THEY, WHERE.NORTH, WHEN.WINTER),
    (21, WHO.YOU, WHERE.WEST, WHEN.AUTUMN),
    (42, WHO.ME, WHERE.EAST, WHEN.SPRING),
    (63, WHO.WE, WHERE.SOUTH, WHEN.SUMMER),
])
def test_from_index(idx, expected_who, expected_where, expected_when):
    s = SubitState.from_index(idx)
    assert s.who == expected_who
    assert s.where == expected_where
    assert s.when == expected_when

def test_from_index_out_of_range():
    with pytest.raises(ValueError, match="index must be 0..63"):
        SubitState.from_index(-1)
    with pytest.raises(ValueError, match="index must be 0..63"):
        SubitState.from_index(64)


# ---------- Test from_names and to_names ----------
def test_from_names():
    s = SubitState.from_names("WE", "EAST", "SUMMER")
    assert s.who == WHO.WE
    assert s.where == WHERE.EAST
    assert s.when == WHEN.SUMMER

def test_to_names():
    s = SubitState(who=WHO.ME, where=WHERE.SOUTH, when=WHEN.AUTUMN)
    assert s.to_names() == ("ME", "SOUTH", "AUTUMN")

def test_str():
    s = SubitState(who=WHO.WE, where=WHERE.EAST, when=WHEN.SUMMER)
    assert str(s) == "WE/EAST/SUMMER"

def test_repr():
    s = SubitState(who=WHO.ME, where=WHERE.NORTH, when=WHEN.WINTER)
    assert repr(s) == "SubitState(who=2, where=0, when=0)"


# ---------- Test all_states ----------
def test_all_states_length():
    states = all_states()
    assert len(states) == 64
    # Check that indices are 0..63 in order
    for i, state in enumerate(states):
        assert state.to_index() == i


# ---------- Test helper functions ----------
def test_index_to_triplet():
    assert index_to_triplet(0) == (0, 0, 0)
    assert index_to_triplet(21) == (1, 1, 1)   # 0b010101
    assert index_to_triplet(42) == (2, 2, 2)   # 0b101010
    assert index_to_triplet(63) == (3, 3, 3)

def test_triplet_to_index():
    assert triplet_to_index(0, 0, 0) == 0
    assert triplet_to_index(1, 1, 1) == 21
    assert triplet_to_index(2, 2, 2) == 42
    assert triplet_to_index(3, 3, 3) == 63


# ---------- Test pre‑computed arrays ----------
def test_state_names_length():
    assert len(STATE_NAMES) == 64
    assert STATE_NAMES[0] == "THEY/NORTH/WINTER"
    assert STATE_NAMES[63] == "WE/SOUTH/SUMMER"

def test_who_of_index():
    assert len(WHO_OF_INDEX) == 64
    assert WHO_OF_INDEX[0] == 0
    assert WHO_OF_INDEX[21] == 1
    assert WHO_OF_INDEX[42] == 2
    assert WHO_OF_INDEX[63] == 3

def test_where_of_index():
    assert len(WHERE_OF_INDEX) == 64
    assert WHERE_OF_INDEX[0] == 0
    assert WHERE_OF_INDEX[21] == 1
    assert WHERE_OF_INDEX[42] == 2
    assert WHERE_OF_INDEX[63] == 3

def test_when_of_index():
    assert len(WHEN_OF_INDEX) == 64
    assert WHEN_OF_INDEX[0] == 0
    assert WHEN_OF_INDEX[21] == 1
    assert WHEN_OF_INDEX[42] == 2
    assert WHEN_OF_INDEX[63] == 3


# ---------- Test name dictionaries ----------
def test_who_names():
    assert WHO_NAMES[WHO.THEY] == "THEY"
    assert WHO_NAMES[WHO.YOU] == "YOU"
    assert WHO_NAMES[WHO.ME] == "ME"
    assert WHO_NAMES[WHO.WE] == "WE"

def test_where_names():
    assert WHERE_NAMES[WHERE.NORTH] == "NORTH"
    assert WHERE_NAMES[WHERE.WEST] == "WEST"
    assert WHERE_NAMES[WHERE.EAST] == "EAST"
    assert WHERE_NAMES[WHERE.SOUTH] == "SOUTH"

def test_when_names():
    assert WHEN_NAMES[WHEN.WINTER] == "WINTER"
    assert WHEN_NAMES[WHEN.AUTUMN] == "AUTUMN"
    assert WHEN_NAMES[WHEN.SPRING] == "SPRING"
    assert WHEN_NAMES[WHEN.SUMMER] == "SUMMER"

def test_reverse_name_mappings():
    assert WHO_FROM_NAME["THEY"] == WHO.THEY
    assert WHO_FROM_NAME["YOU"] == WHO.YOU
    assert WHO_FROM_NAME["ME"] == WHO.ME
    assert WHO_FROM_NAME["WE"] == WHO.WE
    assert WHERE_FROM_NAME["NORTH"] == WHERE.NORTH
    assert WHERE_FROM_NAME["WEST"] == WHERE.WEST
    assert WHERE_FROM_NAME["EAST"] == WHERE.EAST
    assert WHERE_FROM_NAME["SOUTH"] == WHERE.SOUTH
    assert WHEN_FROM_NAME["WINTER"] == WHEN.WINTER
    assert WHEN_FROM_NAME["AUTUMN"] == WHEN.AUTUMN
    assert WHEN_FROM_NAME["SPRING"] == WHEN.SPRING
    assert WHEN_FROM_NAME["SUMMER"] == WHEN.SUMMER