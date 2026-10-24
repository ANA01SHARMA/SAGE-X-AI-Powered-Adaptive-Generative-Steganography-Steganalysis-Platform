"""The byte layout that gets hidden inside an image.

A hidden message is stored in two separate blocks, and the split exists to solve a
bootstrapping problem: the payload's pixel positions depend on the key, the key
depends on the salt, and the salt is itself hidden in the image. Something has to be
readable before the key is known.

So::

    bootstrap block   16 bytes   at FIXED positions        the salt (public, not secret)
    payload block     variable   at KEY-DERIVED positions   header + encrypted payload

The receiver reads the bootstrap block first (no key needed), derives the key from it
and the passphrase, then knows where the payload block lives.

Payload block layout::

    magic      4 bytes   b"SGX1", so we can tell "no message here" from "wrong key"
    version    1 byte    format version
    flags      1 byte    bit 0 = payload was zlib-compressed
    body_len   4 bytes   big-endian length of the encrypted body
    body       variable  nonce || ciphertext || GCM tag

The header is passed to AES-GCM as additional authenticated data, so altering it
breaks decryption rather than silently changing how the body is read.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

from . import crypto

MAGIC = b"SGX1"
VERSION = 1

FLAG_COMPRESSED = 0b0000_0001

#: magic + version + flags + body_len
HEADER_BYTES = len(MAGIC) + 1 + 1 + 4
_HEADER_STRUCT = struct.Struct(">4sBBI")

#: Fixed cost of the payload block alone, excluding the salt. This is what eats into
#: the embedding-rate budget, since the salt lives in its own bootstrap positions.
PAYLOAD_BLOCK_OVERHEAD = HEADER_BYTES + crypto.NONCE_BYTES + crypto.TAG_BYTES

#: Total bytes spent on structure per image, salt included. For reporting.
OVERHEAD_BYTES = crypto.SALT_BYTES + PAYLOAD_BLOCK_OVERHEAD


class FrameError(Exception):
    """The bytes read out of an image are not a valid frame."""


@dataclass(frozen=True)
class Header:
    """Fixed-size preamble of the payload block."""

    version: int
    flags: int
    body_len: int

    def pack(self) -> bytes:
        return _HEADER_STRUCT.pack(MAGIC, self.version, self.flags, self.body_len)

    @classmethod
    def unpack(cls, raw: bytes) -> "Header":
        if len(raw) < HEADER_BYTES:
            raise FrameError("truncated header")
        magic, version, flags, body_len = _HEADER_STRUCT.unpack(raw[:HEADER_BYTES])
        if magic != MAGIC:
            raise FrameError("no SAGE-X message found at these positions")
        if version != VERSION:
            raise FrameError(f"unsupported format version {version}")
        return cls(version=version, flags=flags, body_len=body_len)

    @property
    def compressed(self) -> bool:
        return bool(self.flags & FLAG_COMPRESSED)


def build(payload: bytes, cipher_key: bytes, *, compress: bool = True) -> bytes:
    """Compress, encrypt and frame ``payload`` into the payload block."""
    flags = 0
    body_plain = payload
    if compress:
        squeezed = zlib.compress(payload, level=9)
        # Only keep it if it actually helped, since JPEGs and archives grow.
        if len(squeezed) < len(payload):
            body_plain, flags = squeezed, flags | FLAG_COMPRESSED

    body_len = crypto.sealed_size(len(body_plain))
    header = Header(version=VERSION, flags=flags, body_len=body_len)
    aad = header.pack()
    return aad + crypto.encrypt(cipher_key, body_plain, aad=aad)


def parse(block: bytes, cipher_key: bytes) -> bytes:
    """Reverse :func:`build`. Returns the original payload.

    Raises:
        FrameError: the block is not a SAGE-X frame.
        crypto.DecryptionError: it is, but the passphrase is wrong or it was altered.
    """
    header = Header.unpack(block)
    body = block[HEADER_BYTES : HEADER_BYTES + header.body_len]
    if len(body) < header.body_len:
        raise FrameError("truncated body: image too small or data corrupted")

    payload = crypto.decrypt(cipher_key, body, aad=block[:HEADER_BYTES])
    return zlib.decompress(payload) if header.compressed else payload


def framed_size(payload_len: int, *, compressed_len: int | None = None) -> int:
    """Size of the payload block for a payload of this length.

    Args:
        compressed_len: actual compressed size if known. Compression is data-dependent,
            so for capacity planning we assume the worst case of no compression.
    """
    effective = payload_len if compressed_len is None else min(payload_len, compressed_len)
    return HEADER_BYTES + crypto.sealed_size(effective)
