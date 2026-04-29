"""Caps with multiple holes in profile."""

from __future__ import annotations

from impression.modeling import loft
from impression.modeling.drawing2d import Profile2D, make_rect


def build():
    outer = make_rect(size=(2.4, 1.2)).outer
    hole_a = make_rect(size=(0.4, 0.4), center=(-0.5, 0.0)).outer
    hole_b = make_rect(size=(0.4, 0.4), center=(0.5, 0.0)).outer
    profile = Profile2D(outer=outer, holes=[hole_a, hole_b])
    return loft(
        [profile, profile],
        start_cap="dome",
        end_cap="dome",
        start_cap_length=0.6,
        end_cap_length=0.6,
        cap_scale_dims="both",
    )


if __name__ == "__main__":
    build()
