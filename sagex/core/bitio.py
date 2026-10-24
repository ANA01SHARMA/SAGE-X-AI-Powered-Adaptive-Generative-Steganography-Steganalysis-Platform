"""Conversion between byte strings and flat bit arrays.

Bits are MSB-first within each byte, so ``b"\\x80"`` becomes ``[1,0,0,0,0,0,0,0]``.
Everything downstream works on ``np.uint8`` arrays of 0/1 rather than Python bools,
because the embedders use them to index NumPy arrays.
"""

from __future__ import annotations

import numpy as np

#: 1-D array, dtype=uint8, every value either 0 or 1.
BitArray = np.ndarray


def bytes_to_bits(data: bytes) -> BitArray:
    """Expand ``data`` into one bit per element, MSB-first."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: BitArray) -> bytes:
    """Pack a bit array back into bytes. Inverse of :func:`bytes_to_bits`.

    Raises:
        ValueError: if ``bits`` is not a whole number of bytes.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    if len(bits) % 8:
        raise ValueError(f"bit count {len(bits)} is not a multiple of 8")
    return np.packbits(bits).tobytes()


def bit_length(n_bytes: int) -> int:
    """Number of bits needed to carry ``n_bytes`` bytes."""
    return n_bytes * 8
