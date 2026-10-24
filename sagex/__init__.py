"""SAGE-X: encrypted data hidden inside images, with an honest capacity check.

Typical use::

    from sagex import hide, reveal, imageio

    cover = imageio.load("photo.png")
    result = hide(cover, b"secret message", passphrase="correct horse")
    imageio.save("carrier.png", result.stego)

    recovered = reveal(imageio.load("carrier.png"), passphrase="correct horse")

Before embedding, ask what will fit::

    from sagex import capacity
    print(capacity.analyse(cover).summary())
"""

from . import capacity, imageio
from .capacity import Capacity, InsufficientCapacity
from .core.crypto import DecryptionError
from .core.frame import FrameError
from .pipeline import HideResult, hide, hide_file, psnr, reveal, reveal_file

__version__ = "0.1.0"

__all__ = [
    "Capacity",
    "DecryptionError",
    "FrameError",
    "HideResult",
    "InsufficientCapacity",
    "capacity",
    "hide",
    "hide_file",
    "imageio",
    "psnr",
    "reveal",
    "reveal_file",
]
