"""End-to-end tests for the hide/reveal loop."""

from __future__ import annotations

import numpy as np
import pytest

from sagex import capacity, hide, imageio, reveal
from sagex.core import crypto, frame

PASSPHRASE = "correct horse battery staple"


@pytest.fixture
def cover() -> np.ndarray:
    """A textured 256x256 colour image.

    Deterministic, and noisy enough to be a realistic cover - a flat image would
    make the capacity numbers unrepresentative.
    """
    rng = np.random.default_rng(1234)
    base = rng.integers(40, 215, size=(256, 256, 3), dtype=np.uint8)
    # Add smooth structure so it is not pure noise.
    gradient = np.linspace(0, 40, 256, dtype=np.uint8)
    return np.clip(base + gradient[None, :, None], 0, 255).astype(np.uint8)


@pytest.fixture
def small_cover() -> np.ndarray:
    """Too small to carry anything once overhead is counted."""
    return np.full((24, 24, 3), 128, dtype=np.uint8)


# --------------------------------------------------------------------- round trip

def test_roundtrip_returns_payload_unchanged(cover):
    payload = b"meet me at the usual place, 7pm"
    result = hide(cover, payload, PASSPHRASE)
    assert reveal(result.stego, PASSPHRASE) == payload


def test_roundtrip_handles_binary_payload(cover):
    payload = bytes(range(256)) * 4
    result = hide(cover, payload, PASSPHRASE)
    assert reveal(result.stego, PASSPHRASE) == payload


def test_roundtrip_handles_empty_payload(cover):
    result = hide(cover, b"", PASSPHRASE)
    assert reveal(result.stego, PASSPHRASE) == b""


def test_roundtrip_preserves_grayscale(cover):
    gray = imageio.to_luminance(cover)
    payload = b"grayscale works too"
    result = hide(gray, payload, PASSPHRASE)
    assert result.stego.shape == gray.shape
    assert reveal(result.stego, PASSPHRASE) == payload


def test_stego_is_visually_identical(cover):
    result = hide(cover, b"x" * 200, PASSPHRASE)
    assert result.psnr_db > 50, "embedding should be imperceptible"
    assert result.stego.dtype == np.uint8
    assert result.stego.shape == cover.shape


def test_changes_are_plus_or_minus_one_only(cover):
    """LSB matching must never move a pixel by more than 1."""
    result = hide(cover, b"y" * 300, PASSPHRASE)
    delta = result.stego.astype(int) - cover.astype(int)
    assert set(np.unique(delta)).issubset({-1, 0, 1})


# ------------------------------------------------------------------------ security

def test_wrong_passphrase_is_rejected(cover):
    result = hide(cover, b"top secret", PASSPHRASE)
    # A wrong key points at different pixels, so this usually fails to find the
    # magic bytes; occasionally it finds them and fails authentication instead.
    with pytest.raises((frame.FrameError, crypto.DecryptionError)):
        reveal(result.stego, "wrong passphrase")


def test_tampering_is_detected(cover):
    result = hide(cover, b"authentic message", PASSPHRASE)
    tampered = result.stego.copy()
    tampered[100:140, 100:140] ^= 1  # flip low bits over a patch

    with pytest.raises((frame.FrameError, crypto.DecryptionError)):
        reveal(tampered, PASSPHRASE)


def test_clean_image_reports_no_message(cover):
    with pytest.raises(frame.FrameError):
        reveal(cover, PASSPHRASE)


def test_same_payload_twice_gives_different_images(cover):
    """Fresh salt and nonce each time, so identical inputs must not collide."""
    first = hide(cover, b"same message", PASSPHRASE)
    second = hide(cover, b"same message", PASSPHRASE)
    assert not np.array_equal(first.stego, second.stego)


# ------------------------------------------------------------------------ capacity

def test_refuses_oversized_payload(cover):
    report = capacity.analyse(cover)
    with pytest.raises(capacity.InsufficientCapacity) as excinfo:
        hide(cover, b"z" * (report.max_payload_bytes + 1), PASSPHRASE)
    assert "holds at most" in str(excinfo.value)


def test_accepts_payload_at_exactly_the_limit(cover):
    report = capacity.analyse(cover)
    payload = bytes(np.random.default_rng(7).integers(0, 256, report.max_payload_bytes,
                                                      dtype=np.uint8))
    result = hide(cover, payload, PASSPHRASE)
    assert reveal(result.stego, PASSPHRASE) == payload


def test_tiny_image_reports_zero_capacity(small_cover):
    report = capacity.analyse(small_cover)
    assert report.max_payload_bytes == 0
    assert not report.is_usable
    assert "cannot carry a message" in report.summary()


def test_tiny_image_refuses_rather_than_failing_later(small_cover):
    with pytest.raises(capacity.InsufficientCapacity):
        hide(small_cover, b"hi", PASSPHRASE)


def test_higher_rate_gives_more_capacity(cover):
    conservative = capacity.analyse(cover, capacity.RATE_CONSERVATIVE)
    aggressive = capacity.analyse(cover, capacity.RATE_AGGRESSIVE)
    assert aggressive.max_payload_bytes > conservative.max_payload_bytes


def test_overhead_constants_are_consistent():
    assert frame.OVERHEAD_BYTES == crypto.SALT_BYTES + frame.PAYLOAD_BLOCK_OVERHEAD
    assert frame.PAYLOAD_BLOCK_OVERHEAD == (
        frame.HEADER_BYTES + crypto.NONCE_BYTES + crypto.TAG_BYTES
    )
