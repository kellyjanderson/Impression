from __future__ import annotations

from typing import Sequence

import numpy as np

from ._profile2d import _loops_resampled
from .drawing2d import Path2D, Profile2D


def morph_profiles(
    start: Profile2D,
    end: Profile2D,
    t: float,
    samples: int = 200,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> Profile2D:
    """Interpolate between two profiles with matching topology."""

    t = float(t)
    if t < 0.0 or t > 1.0:
        raise ValueError("t must be in [0, 1].")
    if len(start.holes) != len(end.holes):
        raise ValueError("Profiles must have the same number of holes to morph.")

    loops_a = _loops_resampled(start, samples, segments_per_circle, bezier_samples)
    loops_b = _loops_resampled(end, samples, segments_per_circle, bezier_samples)

    blended_loops = []
    for a, b in zip(loops_a, loops_b):
        blended_loops.append((1.0 - t) * a + t * b)

    outer = Path2D.from_points(blended_loops[0], closed=True)
    holes = [Path2D.from_points(loop, closed=True) for loop in blended_loops[1:]]
    profile = Profile2D(outer=outer, holes=holes)

    if start.color is not None and end.color is not None:
        profile.color = tuple((1.0 - t) * np.array(start.color) + t * np.array(end.color))
    elif start.color is not None:
        profile.color = start.color
    elif end.color is not None:
        profile.color = end.color
    return profile


def morph(
    start: Profile2D,
    end: Profile2D,
    t: float,
    samples: int = 200,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> Profile2D:
    """Alias for morph_profiles."""

    return morph_profiles(
        start=start,
        end=end,
        t=t,
        samples=samples,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    )


__all__ = ["morph_profiles", "morph"]
