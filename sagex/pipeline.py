"""The complete loop: hide a payload in an image, get it back byte-for-byte.

    payload -> compress -> encrypt -> frame -> scatter across pixels -> PNG
    PNG -> read salt -> derive key -> locate payload -> decrypt -> payload

Reading the salt before the key exists is why there are two position sets. See
:mod:`sagex.core.frame` for the layout and why it is shaped that way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import capacity, imageio
from .core import crypto, frame
from .core.bitio import bits_to_bytes, bytes_to_bits
from .embed import lsbm, positions


@dataclass(frozen=True)
class HideResult:
    """A stego image plus what it cost to make it."""

    stego: np.ndarray
    payload_bytes: int
    stored_bytes: int
    pixels_changed: int
    psnr_db: float
    capacity: capacity.Capacity

    @property
    def compression_ratio(self) -> float:
        """Stored size over original size. Below 1.0 means compression helped."""
        return self.stored_bytes / self.payload_bytes if self.payload_bytes else 1.0

    def summary(self) -> str:
        return (
            f"Hid {self.payload_bytes} bytes (stored as {self.stored_bytes} after "
            f"compression and encryption) by changing {self.pixels_changed:,} of "
            f"{self.capacity.total_slots:,} values. Image quality {self.psnr_db:.1f} dB, "
            f"which is visually identical."
        )


def hide(
    cover: np.ndarray,
    payload: bytes,
    passphrase: str,
    *,
    rate: float = capacity.DEFAULT_RATE,
    rng: np.random.Generator | None = None,
) -> HideResult:
    """Hide ``payload`` inside ``cover``.

    Raises:
        InsufficientCapacity: the payload will not fit. Checked *before* any work,
            so we never hand back an image that cannot be read again.
    """
    report = capacity.require_fit(cover, len(payload), rate)

    salt = crypto.new_salt()
    keys = crypto.derive_keys(passphrase, salt)
    block = frame.build(payload, keys.cipher_key)

    flat = cover.ravel()
    n_slots = flat.size
    rng = rng or np.random.default_rng()

    # Block 1: the salt, at fixed positions any receiver can find.
    stego_flat = lsbm.embed_bits(
        flat, bytes_to_bits(salt), positions.bootstrap_positions(n_slots), rng
    )

    # Block 2: the encrypted payload, at positions only the key reveals.
    block_bits = bytes_to_bits(block)
    stego_flat = lsbm.embed_bits(
        stego_flat,
        block_bits,
        positions.payload_positions(n_slots, keys.perm_seed, len(block_bits)),
        rng,
    )

    stego = stego_flat.reshape(cover.shape)
    return HideResult(
        stego=stego,
        payload_bytes=len(payload),
        stored_bytes=len(block),
        pixels_changed=lsbm.changes_made(cover, stego),
        psnr_db=psnr(cover, stego),
        capacity=report,
    )


def reveal(stego: np.ndarray, passphrase: str) -> bytes:
    """Recover the payload hidden by :func:`hide`.

    Raises:
        FrameError: no SAGE-X message found at the positions this key points to.
        DecryptionError: a message is there, but the passphrase is wrong or the
            image was altered after embedding.
    """
    flat = stego.ravel()
    n_slots = flat.size

    salt_bits = lsbm.extract_bits(
        flat, positions.bootstrap_positions(n_slots), positions.BOOTSTRAP_BITS
    )
    keys = crypto.derive_keys(passphrase, bits_to_bytes(salt_bits))

    # Read the header first to learn how long the body is, then read the whole block.
    # payload_positions returns a prefix of one deterministic permutation, so asking
    # for more positions extends the same sequence rather than producing a new one.
    header_bits = frame.HEADER_BYTES * 8
    slots = positions.payload_positions(n_slots, keys.perm_seed, header_bits)
    header = frame.Header.unpack(bits_to_bytes(lsbm.extract_bits(flat, slots, header_bits)))

    total_bits = (frame.HEADER_BYTES + header.body_len) * 8
    if total_bits > n_slots - positions.BOOTSTRAP_BITS:
        raise frame.FrameError("declared payload is larger than the image can hold")

    slots = positions.payload_positions(n_slots, keys.perm_seed, total_bits)
    block = bits_to_bytes(lsbm.extract_bits(flat, slots, total_bits))
    return frame.parse(block, keys.cipher_key)


def hide_file(
    cover_path: str, payload_path: str, out_path: str, passphrase: str,
    *, rate: float = capacity.DEFAULT_RATE,
) -> HideResult:
    """Convenience wrapper: read a cover and a file, write a stego PNG."""
    cover = imageio.load(cover_path)
    with open(payload_path, "rb") as handle:
        payload = handle.read()
    result = hide(cover, payload, passphrase, rate=rate)
    imageio.save(out_path, result.stego)
    return result


def reveal_file(stego_path: str, out_path: str, passphrase: str) -> int:
    """Convenience wrapper: read a stego PNG, write the recovered payload."""
    payload = reveal(imageio.load(stego_path), passphrase)
    with open(out_path, "wb") as handle:
        handle.write(payload)
    return len(payload)


def psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB. Above ~40 dB is visually indistinguishable.

    The difference is taken in int16 and accumulated with einsum, which never
    materialises the squared array. Promoting two 12-megapixel images to float64
    would allocate over half a gigabyte to produce one number.
    """
    delta = np.subtract(original, modified, dtype=np.int16).ravel()
    total = float(np.einsum("i,i->", delta, delta, dtype=np.int64))
    if total == 0:
        return float("inf")
    return 10.0 * float(np.log10(255.0**2 / (total / delta.size)))
