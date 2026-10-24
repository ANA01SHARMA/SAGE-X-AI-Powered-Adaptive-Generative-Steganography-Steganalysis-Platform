"""Loading and saving carrier images.

Everything inside SAGE-X works on ``uint8`` NumPy arrays: ``(H, W)`` for grayscale,
``(H, W, 3)`` for colour.

Saving is PNG-only and deliberately so. A stego image must arrive byte-identical -
JPEG or WebP would re-quantise the pixels and destroy every hidden bit. Refusing a
lossy extension here is cheaper than debugging silent data loss later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

LOSSLESS_SUFFIXES = {".png", ".bmp", ".tiff", ".tif"}


class LossyFormatError(Exception):
    """Asked to save a stego image in a format that would destroy the payload."""


def load(path: str | Path, *, grayscale: bool = False) -> np.ndarray:
    """Read an image as a uint8 array."""
    with Image.open(path) as handle:
        converted = handle.convert("L" if grayscale else "RGB")
        return np.asarray(converted, dtype=np.uint8)


def save(path: str | Path, image: np.ndarray) -> None:
    """Write a uint8 array losslessly.

    Raises:
        LossyFormatError: if the extension implies a lossy codec.
    """
    path = Path(path)
    if path.suffix.lower() not in LOSSLESS_SUFFIXES:
        raise LossyFormatError(
            f"{path.suffix or 'no extension'} is a lossy format and would destroy the "
            f"hidden data. Use one of: {', '.join(sorted(LOSSLESS_SUFFIXES))}"
        )
    Image.fromarray(image).save(path)


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Collapse to a 2-D grayscale view, for cost analysis.

    Colour images are analysed on luminance because that is where the human eye and
    most detectors are most sensitive.
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return np.asarray(Image.fromarray(image).convert("L"), dtype=np.uint8)
    raise ValueError(f"unsupported image shape {image.shape}")


def describe(image: np.ndarray) -> str:
    """Human-readable shape summary, for reports and CLI output."""
    if image.ndim == 2:
        height, width = image.shape
        return f"{width}x{height} grayscale"
    height, width, channels = image.shape
    return f"{width}x{height} x{channels} colour"
