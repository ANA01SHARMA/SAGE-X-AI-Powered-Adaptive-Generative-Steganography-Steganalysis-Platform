"""Tests for pixel-position selection.

The prefix property is the important one. The extractor reads the frame header
before it knows how long the body is, so it asks for a few positions and then for
many more - and the second answer must begin with the first.
"""

from __future__ import annotations

import numpy as np
import pytest

from sagex.embed import positions

SEED = bytes(range(32))
OTHER_SEED = bytes(range(1, 33))

# (carrier slots, a payload size that fits)
SIZES = [(3072, 2000), (786_432, 8000), (12_000_000, 50_000)]


@pytest.mark.parametrize("n_slots,count", SIZES)
def test_positions_are_a_stable_prefix(n_slots, count):
    """Asking for more positions must extend the sequence, not replace it."""
    short = positions.payload_positions(n_slots, SEED, 80)
    long = positions.payload_positions(n_slots, SEED, count)
    assert np.array_equal(short, long[:80])


@pytest.mark.parametrize("n_slots,count", SIZES)
def test_positions_are_distinct(n_slots, count):
    chosen = positions.payload_positions(n_slots, SEED, count)
    assert chosen.size == count
    assert len(np.unique(chosen)) == count


@pytest.mark.parametrize("n_slots,count", SIZES)
def test_positions_stay_in_range(n_slots, count):
    chosen = positions.payload_positions(n_slots, SEED, count)
    assert chosen.min() >= 0
    assert chosen.max() < n_slots


@pytest.mark.parametrize("n_slots,count", SIZES)
def test_payload_never_collides_with_bootstrap(n_slots, count):
    """The salt must not be overwritten by payload bits."""
    reserved = set(positions.bootstrap_positions(n_slots).tolist())
    chosen = set(positions.payload_positions(n_slots, SEED, count).tolist())
    assert not (reserved & chosen)


def test_positions_are_reproducible():
    a = positions.payload_positions(786_432, SEED, 5000)
    b = positions.payload_positions(786_432, SEED, 5000)
    assert np.array_equal(a, b)


def test_different_keys_give_different_positions():
    a = positions.payload_positions(786_432, SEED, 5000)
    b = positions.payload_positions(786_432, OTHER_SEED, 5000)
    assert not np.array_equal(a, b)


def test_positions_depend_on_carrier_size():
    """A resized image must fail cleanly rather than half-decode."""
    a = positions.payload_positions(786_432, SEED, 1000)
    b = positions.payload_positions(786_433, SEED, 1000)
    assert not np.array_equal(a, b)


def test_bootstrap_is_reproducible_without_a_key():
    a = positions.bootstrap_positions(786_432)
    b = positions.bootstrap_positions(786_432)
    assert np.array_equal(a, b)
    assert a.size == positions.BOOTSTRAP_BITS


def test_saturated_carrier_still_fills():
    """Requesting every available slot must succeed, not exhaust the retry budget."""
    n_slots = 3072
    available = n_slots - positions.BOOTSTRAP_BITS
    chosen = positions.payload_positions(n_slots, SEED, available)
    assert chosen.size == available
    assert len(np.unique(chosen)) == available


def test_overfull_request_is_refused():
    with pytest.raises(ValueError, match="only"):
        positions.payload_positions(3072, SEED, 3072)


def test_zero_count_returns_empty():
    assert positions.payload_positions(786_432, SEED, 0).size == 0


def test_carrier_too_small_for_bootstrap_is_refused():
    with pytest.raises(ValueError, match="at least"):
        positions.bootstrap_positions(positions.BOOTSTRAP_BITS - 1)
