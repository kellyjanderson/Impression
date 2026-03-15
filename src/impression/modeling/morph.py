from __future__ import annotations

from typing import Sequence

import numpy as np

from .topology import Loop, Region, Section, as_section, resample_loop


def morph_profiles(
    start: Section | Region | object,
    end: Section | Region | object,
    t: float,
    samples: int = 200,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> Section:
    """Interpolate between two planar sections with matching topology."""

    t = float(t)
    if t < 0.0 or t > 1.0:
        raise ValueError("t must be in [0, 1].")
    start_section = as_section(
        start,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    ).normalized()
    end_section = as_section(
        end,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    ).normalized()
    if len(start_section.regions) != len(end_section.regions):
        raise ValueError("Sections must have the same number of regions to morph.")

    result_regions: list[Region] = []
    for start_region, end_region in zip(start_section.regions, end_section.regions, strict=True):
        if len(start_region.holes) != len(end_region.holes):
            raise ValueError("Regions must have the same number of holes to morph.")
        start_loops = [start_region.outer.points, *(hole.points for hole in start_region.holes)]
        end_loops = [end_region.outer.points, *(hole.points for hole in end_region.holes)]

        blended_loops: list[np.ndarray] = []
        for a, b in zip(start_loops, end_loops, strict=True):
            a_resampled = resample_loop(a, samples)
            b_resampled = resample_loop(b, samples)
            blended_loops.append((1.0 - t) * a_resampled + t * b_resampled)

        outer = Loop(np.asarray(blended_loops[0], dtype=float))
        holes = tuple(Loop(np.asarray(loop, dtype=float)) for loop in blended_loops[1:])
        result_regions.append(Region(outer=outer, holes=holes).normalized())

    return Section(tuple(result_regions)).normalized()


def morph(
    start: Section | Region | object,
    end: Section | Region | object,
    t: float,
    samples: int = 200,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> Section:
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
