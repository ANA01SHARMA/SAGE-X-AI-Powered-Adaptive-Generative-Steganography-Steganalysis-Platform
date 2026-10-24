"""Choosing which pixels carry which bits.

Two position sets, for the two blocks described in :mod:`sagex.core.frame`:

:func:`bootstrap_positions`
    Fixed and public. Derived from the image dimensions alone, so a receiver with no
    key can still find the salt. Only 128 pixels, so the statistical footprint is nil.

:func:`payload_positions`
    Derived from the key. Without the passphrase an attacker does not know which
    pixels to even look at, and the bootstrap pixels are excluded so the two blocks
    cannot collide.
"""

from __future__ import annotations

import numpy as np

from ..core.crypto import SALT_BYTES, stable_digest

#: How many pixels the bootstrap block occupies (one bit each).
BOOTSTRAP_BITS = SALT_BYTES * 8

_BOOTSTRAP_DOMAIN = b"SAGEX-bootstrap-v1"
_PAYLOAD_DOMAIN = b"SAGEX-payload-v1"


def _rng_from(*parts: bytes) -> np.random.Generator:
    """A reproducible generator seeded by the given byte strings."""
    return np.random.default_rng(np.frombuffer(stable_digest(*parts), dtype=np.uint32))


def _shape_tag(n_slots: int) -> bytes:
    """Bind positions to the carrier size, so a resized image never half-decodes."""
    return str(n_slots).encode("ascii")


def bootstrap_positions(n_slots: int) -> np.ndarray:
    """Fixed positions carrying the salt. Identical for sender and receiver."""
    if n_slots < BOOTSTRAP_BITS:
        raise ValueError(f"carrier has {n_slots} slots, needs at least {BOOTSTRAP_BITS}")
    rng = _rng_from(_BOOTSTRAP_DOMAIN, _shape_tag(n_slots))
    return rng.choice(n_slots, size=BOOTSTRAP_BITS, replace=False)


#: Positions are drawn in fixed-size chunks so that the sequence depends only on the
#: key and the carrier size, never on how many positions were asked for. That is what
#: makes the result a stable *prefix*, so asking for 80 positions and then for 8,000
#: gives the same first 80. The extractor relies on this, reading the header before it
#: knows how long the body is.
_CHUNK = 1 << 16


def payload_positions(n_slots: int, perm_seed: bytes, count: int) -> np.ndarray:
    """Key-derived positions for ``count`` payload bits.

    Draws random indices in chunks and discards repeats, rather than permuting the
    whole carrier. Cost scales with the payload, not the image: picking a few thousand
    positions out of a 12-megapixel photo takes milliseconds instead of seconds, and
    allocates kilobytes instead of 275 MB.
    """
    available = n_slots - BOOTSTRAP_BITS
    if count > available:
        raise ValueError(f"need {count} slots, only {available} available")
    if count == 0:
        return np.empty(0, dtype=np.int64)

    reserved = bootstrap_positions(n_slots)
    rng = _rng_from(_PAYLOAD_DOMAIN, perm_seed, _shape_tag(n_slots))

    chunk = min(_CHUNK, n_slots)
    picked = np.empty(0, dtype=np.int64)
    # Generous bound: each round claims most of what it draws, so this only trips if
    # `count` approaches `available` on a very small carrier.
    for _ in range(64):
        if picked.size >= count:
            break
        batch = rng.integers(0, n_slots, size=chunk, dtype=np.int64)
        batch = batch[~np.isin(batch, reserved)]
        # Drop repeats inside this batch, keeping the order they were drawn in.
        _, first_seen = np.unique(batch, return_index=True)
        batch = batch[np.sort(first_seen)]
        if picked.size:
            batch = batch[~np.isin(batch, picked)]
        picked = np.concatenate([picked, batch])
    else:
        raise ValueError(
            f"could not place {count} bits in a {n_slots}-slot carrier; it is too full"
        )

    return picked[:count]
