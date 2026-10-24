"""LSB matching: the actual bit-hiding step.

To store a bit in a pixel we make the pixel's lowest bit equal to it. If it already
matches, nothing changes. If not, we add or subtract 1 at random.

Adding or subtracting, rather than overwriting the low bit, is what distinguishes
*LSB matching* from plain *LSB replacement*. Replacement pairs values (254 only ever
becomes 255, never 253), which creates a statistical signature that detectors from
the early 2000s already pick up. Matching leaves no such pairing.

The receiver only reads the low bit, so it never needs to know which direction we
chose.

Limitation (v1):
    Positions come from the key alone, not from the HILL cost map, so embedding is
    not yet content-adaptive. Steering by cost requires syndrome-trellis coding,
    because a receiver recomputing cost from the stego image would rank pixels
    slightly differently from the sender and lose sync. STC sidesteps this by letting
    the receiver read a syndrome from fixed positions while the sender chooses freely
    which pixels to change. That lands in ``stc.py``; this module's interface is
    designed so it can be swapped in without touching callers.
"""

from __future__ import annotations

import numpy as np

from ..core.bitio import BitArray

_PIXEL_MIN = 0
_PIXEL_MAX = 255


def embed_bits(
    carrier: np.ndarray,
    bits: BitArray,
    positions: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a copy of ``carrier`` with ``bits`` stored at ``positions``.

    Args:
        carrier: flat uint8 array of pixel values.
        bits: the bits to store, one per position.
        positions: indices into ``carrier``; must be at least as long as ``bits``.
        rng: source for the +1/-1 choice. Defaults to fresh randomness.

    Returns:
        A new uint8 array. The input is not modified.
    """
    if len(positions) < len(bits):
        raise ValueError(f"{len(bits)} bits need {len(bits)} positions, got {len(positions)}")

    rng = rng or np.random.default_rng()
    stego = carrier.copy()
    slots = positions[: len(bits)]

    values = stego[slots].astype(np.int16)
    needs_change = (values & 1) != bits

    # Random direction, then fix up the two values that can only move one way.
    step = rng.choice([-1, 1], size=len(values)).astype(np.int16)
    step[values <= _PIXEL_MIN] = 1
    step[values >= _PIXEL_MAX] = -1

    stego[slots] = np.where(needs_change, values + step, values).astype(np.uint8)
    return stego


def extract_bits(carrier: np.ndarray, positions: np.ndarray, count: int) -> BitArray:
    """Read ``count`` bits back out of ``carrier``."""
    if len(positions) < count:
        raise ValueError(f"need {count} positions, got {len(positions)}")
    return (carrier[positions[:count]] & 1).astype(np.uint8)


def changes_made(cover: np.ndarray, stego: np.ndarray) -> int:
    """How many pixels differ. Useful for reporting and tests."""
    return int(np.count_nonzero(cover != stego))
