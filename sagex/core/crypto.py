"""Encryption and key derivation.

Two secrets are derived from one passphrase:

``cipher_key``
    Encrypts the payload with AES-256-GCM, which also authenticates it. A wrong
    passphrase, or a modified image, fails loudly instead of returning garbage.

``perm_seed``
    Chooses *which* pixels carry the payload. Without the passphrase an attacker
    does not know where to look, so the bits are not merely encrypted but scattered.

Both come from Argon2id, which is deliberately slow and memory-hard so that guessing
passphrases costs real time and RAM.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from argon2.low_level import Type, hash_secret_raw
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SALT_BYTES = 16
NONCE_BYTES = 12
TAG_BYTES = 16
KEY_BYTES = 32

# Argon2id cost, tuned to roughly 0.5 s on a laptop CPU. Slow enough to punish
# guessing, fast enough that a user does not think the app has hung.
_TIME_COST = 3
_MEMORY_COST_KIB = 64 * 1024  # 64 MiB
_PARALLELISM = 2


class DecryptionError(Exception):
    """Wrong passphrase, or the image was altered after embedding."""


@dataclass(frozen=True)
class Keys:
    """Everything derived from one passphrase and one salt."""

    cipher_key: bytes  # AES-256-GCM key
    perm_seed: bytes  # seeds the pixel-position permutation


def derive_keys(passphrase: str, salt: bytes) -> Keys:
    """Stretch ``passphrase`` into a cipher key and a permutation seed.

    The salt is public, since it travels in the image. It exists so that the same
    passphrase produces different keys and different pixel positions for every
    image. Reusing one permutation across many images would let an attacker who
    collects several of them correlate the positions.
    """
    if len(salt) != SALT_BYTES:
        raise ValueError(f"salt must be {SALT_BYTES} bytes, got {len(salt)}")

    # One Argon2id pass yields 64 bytes, split into the two independent secrets.
    material = hash_secret_raw(
        secret=passphrase.encode("utf-8"),
        salt=salt,
        time_cost=_TIME_COST,
        memory_cost=_MEMORY_COST_KIB,
        parallelism=_PARALLELISM,
        hash_len=KEY_BYTES * 2,
        type=Type.ID,
    )
    return Keys(cipher_key=material[:KEY_BYTES], perm_seed=material[KEY_BYTES:])


def new_salt() -> bytes:
    """A fresh random salt for a new embedding."""
    return os.urandom(SALT_BYTES)


def encrypt(key: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
    """Encrypt with AES-256-GCM.

    Returns ``nonce || ciphertext || tag``. The nonce is random per call and must
    never repeat under one key, because GCM loses all confidentiality if it does.

    Args:
        aad: additional authenticated data. Not encrypted, but covered by the tag,
            so we bind the frame header to the ciphertext and tampering is detected.
    """
    nonce = os.urandom(NONCE_BYTES)
    sealed = AESGCM(key).encrypt(nonce, plaintext, aad)
    return nonce + sealed


def decrypt(key: bytes, blob: bytes, aad: bytes = b"") -> bytes:
    """Reverse :func:`encrypt`.

    Raises:
        DecryptionError: wrong passphrase, or the data was modified.
    """
    if len(blob) < NONCE_BYTES + TAG_BYTES:
        raise DecryptionError("ciphertext too short to be valid")
    nonce, sealed = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
    try:
        return AESGCM(key).decrypt(nonce, sealed, aad)
    except InvalidTag as exc:
        raise DecryptionError(
            "authentication failed: wrong passphrase, or the image was altered"
        ) from exc


def sealed_size(plaintext_len: int) -> int:
    """Bytes that :func:`encrypt` will produce for a payload of this length."""
    return NONCE_BYTES + plaintext_len + TAG_BYTES


def stable_digest(*parts: bytes) -> bytes:
    """A deterministic 32-byte digest, used where we need a reproducible seed."""
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(4, "big"))
        h.update(part)
    return h.digest()
