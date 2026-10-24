"""How much can this image actually hold?

This is the part other steganography tools do not do. Steghide, OpenStego and
SilentEye embed whatever you hand them and report success; you discover the failure
only when extraction returns nothing.

We work the answer out first, and refuse when the answer is "nothing".

Capacity has three limits stacked on top of each other:

1. **Physical**: one bit per colour channel per pixel.
2. **Safety**: we only use a fraction of those, set by the embedding rate (bpp).
   Using every pixel would be trivially detectable.
3. **Overhead**: the salt, header, nonce and authentication tag cost 54 bytes
   before any payload fits at all.

Small images fail at step 3: a 512x512 image at a conservative rate has room for
fewer bytes than the overhead needs, so the honest answer is zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import frame
from .embed import costs, positions
from . import imageio


class InsufficientCapacity(Exception):
    """The payload cannot fit in this image at this embedding rate."""


# Embedding rates, in bits per channel-slot. Lower is safer.
# Grounded in measurement: adaptive embedding stays statistically invisible to a
# trained CNN detector at ~0.05-0.1, and becomes clearly detectable by 0.4.
RATE_CONSERVATIVE = 0.05
RATE_BALANCED = 0.10
RATE_AGGRESSIVE = 0.40

DEFAULT_RATE = RATE_BALANCED

RATE_NAMES = {
    RATE_CONSERVATIVE: "conservative",
    RATE_BALANCED: "balanced",
    RATE_AGGRESSIVE: "aggressive",
}


@dataclass(frozen=True)
class Capacity:
    """What an image can carry, and how comfortable it is doing so."""

    description: str
    total_slots: int
    usable_slots: int
    max_payload_bytes: int
    rate: float
    #: None when scoring was skipped. It costs a full HILL pass and is not needed
    #: to decide whether a payload fits.
    texture_score: float | None

    @property
    def rate_name(self) -> str:
        return RATE_NAMES.get(self.rate, f"{self.rate:g} bpp")

    @property
    def is_usable(self) -> bool:
        return self.max_payload_bytes > 0

    def fits(self, payload_bytes: int) -> bool:
        return payload_bytes <= self.max_payload_bytes

    def summary(self) -> str:
        """One-paragraph explanation, for the CLI and the web UI."""
        if not self.is_usable:
            return (
                f"{self.description}: cannot carry a message at the {self.rate_name} "
                f"rate. {frame.OVERHEAD_BYTES} bytes of encryption overhead are needed "
                f"before any payload fits, and only {self.usable_slots // 8} bytes are "
                f"available. Use a larger image or a higher rate."
            )
        headline = (
            f"{self.description}: holds up to {_human_bytes(self.max_payload_bytes)} "
            f"at the {self.rate_name} rate."
        )
        if self.texture_score is None:
            return headline

        # Thresholds calibrated against the measured range in sagex.embed.costs:
        # detailed photographs land above 0.5, hazy scenes near 0.3, flat below 0.2.
        quality = (
            "richly textured, hides data well"
            if self.texture_score > 0.50
            else "moderately textured"
            if self.texture_score > 0.25
            else "quite smooth, so changes here are easier to spot"
        )
        return f"{headline} Image is {quality} (texture score {self.texture_score:.2f})."


def analyse(image: np.ndarray, rate: float = DEFAULT_RATE,
            *, texture: float | None = None, score_texture: bool = True) -> Capacity:
    """Work out what ``image`` can carry at the given embedding rate.

    Args:
        texture: a texture score already computed for this image. Scoring costs a full
            HILL pass (~0.5 s on a 12-megapixel photo) and does not depend on the rate,
            so callers reporting several rates for one image should compute it once and
            pass it in.
        score_texture: set False to skip scoring entirely when only the byte limit is
            wanted. Whether a payload fits does not depend on texture.
    """
    if not 0 < rate <= 1:
        raise ValueError(f"rate must be in (0, 1], got {rate}")

    total_slots = int(image.size)
    # The bootstrap block carrying the salt comes off the top of the budget; the
    # remaining slots are what the payload block gets.
    budget = int(total_slots * rate) - positions.BOOTSTRAP_BITS
    usable_slots = max(budget, 0)

    # The salt is already accounted for above, so only the payload block's own
    # header, nonce and tag are deducted here.
    payload_bytes = max(usable_slots // 8 - frame.PAYLOAD_BLOCK_OVERHEAD, 0)

    return Capacity(
        description=imageio.describe(image),
        total_slots=total_slots,
        usable_slots=usable_slots,
        max_payload_bytes=payload_bytes,
        rate=rate,
        texture_score=_resolve_texture(image, texture, score_texture),
    )


def _resolve_texture(image: np.ndarray, given: float | None, wanted: bool) -> float | None:
    if given is not None:
        return given
    return costs.texture_score(imageio.to_luminance(image)) if wanted else None


def require_fit(image: np.ndarray, payload_bytes: int, rate: float = DEFAULT_RATE) -> Capacity:
    """Check before embedding. Raise with a useful message if it will not work.

    This is the refusal that makes the tool honest.

    Texture scoring is skipped: whether a payload fits is a question of byte counts,
    and a full HILL pass on a large cover would add most of a second to every embed.
    """
    capacity = analyse(image, rate, score_texture=False)
    if not capacity.fits(payload_bytes):
        raise InsufficientCapacity(
            f"payload is {_human_bytes(payload_bytes)} but this image holds at most "
            f"{_human_bytes(capacity.max_payload_bytes)} at the {capacity.rate_name} "
            f"rate.\n{capacity.summary()}"
        )
    return capacity


def rank_covers(images: dict[str, np.ndarray], payload_bytes: int,
                rate: float = DEFAULT_RATE) -> list[tuple[str, Capacity]]:
    """Sort candidate covers by how comfortably they hold ``payload_bytes``.

    Textured images are preferred over merely large ones. A big photo of a clear sky
    is a worse hiding place than a smaller photo of foliage.
    """
    scored = [(name, analyse(img, rate)) for name, img in images.items()]  # one pass each
    viable = [(n, c) for n, c in scored if c.fits(payload_bytes)]
    # Unscored covers sort last rather than raising: texture_score is optional, and a
    # comparison against None would fail with a confusing TypeError.
    viable.sort(key=lambda pair: (pair[1].texture_score is not None,
                                  pair[1].texture_score or 0.0), reverse=True)
    return viable


def _human_bytes(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count / (1024 * 1024):.2f} MB"
