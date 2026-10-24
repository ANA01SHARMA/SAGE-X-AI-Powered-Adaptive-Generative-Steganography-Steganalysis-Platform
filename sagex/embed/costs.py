"""How noticeable a change to each pixel would be.

Changing a pixel in flat sky is obvious to a statistical detector; changing one in
grass or foliage is not. HILL (Li et al., 2014) turns that intuition into a number
per pixel, where a high cost means "leave this alone".

We use it for two things:

* ranking candidate cover images, since textured photos hide far more than smooth ones
* estimating how many bits an image can safely carry

The embedder itself does not yet steer by cost. Doing so needs syndrome-trellis
coding, because a receiver recomputing cost from the *stego* image would get slightly
different rankings and lose sync. See the note in :mod:`sagex.embed.lsbm`.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, uniform_filter

#: High-pass filter that responds to local detail (the KB filter from the paper).
_HIGH_PASS = np.array(
    [[-1.0, 2.0, -1.0],
     [2.0, -4.0, 2.0],
     [-1.0, 2.0, -1.0]]
)

_EPSILON = 1e-8
_COST_CEILING = 1e8


def hill_cost(image: np.ndarray) -> np.ndarray:
    """Per-pixel embedding cost. Lower means safer to modify.

    Args:
        image: 2-D grayscale array.

    Returns:
        Float array, same shape as ``image``.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {image.shape}")

    detail = convolve(image.astype(np.float64), _HIGH_PASS, mode="mirror")
    # Two averaging passes spread the cost, so isolated safe pixels in an otherwise
    # smooth region do not get used, because those stand out as much as the region does.
    spread = uniform_filter(np.abs(detail), size=3, mode="mirror")
    cost = 1.0 / np.maximum(spread, _EPSILON)
    cost = uniform_filter(cost, size=15, mode="mirror")
    return np.minimum(cost, _COST_CEILING)


#: Median HILL cost of an ideal cover. Measured on uniform random noise, which is
#: the most forgiving carrier there is, since nothing hides better than more noise.
_COST_BEST = 0.003

#: Median cost at which an image is a poor carrier. Measured on real photographs:
#: detailed foliage sits near 0.025, hazy landscapes near 0.35, near-flat scenes
#: above 0.7. A constant image reaches the cost ceiling entirely.
_COST_WORST = 1.0


def texture_score(image: np.ndarray) -> float:
    """A single 0-1 number for how good a cover this image is.

    1.0 means richly textured and safe to embed in; 0 means flat and risky. Used to
    rank candidate covers, so what matters is that it orders images sensibly.

    The scale is logarithmic because measured median costs span three orders of
    magnitude between noise and flat regions; a linear mapping collapses every real
    photograph into the bottom tenth of the range.
    """
    cost = hill_cost(image)
    # Median is the stable summary. The mean is dominated by flat patches, where
    # cost explodes toward the ceiling and swamps everything else.
    median = float(np.median(cost))
    span = np.log10(_COST_WORST) - np.log10(_COST_BEST)
    score = (np.log10(_COST_WORST) - np.log10(max(median, 1e-12))) / span
    return float(np.clip(score, 0.0, 1.0))


def safe_fraction(image: np.ndarray, percentile: float = 40.0) -> float:
    """Fraction of pixels whose cost is low enough to use comfortably.

    Feeds the capacity estimate: a flat image may have millions of pixels but only a
    handful that can be touched without leaving a trace.
    """
    cost = hill_cost(image)
    threshold = np.percentile(cost, percentile)
    return float(np.mean(cost <= threshold))
