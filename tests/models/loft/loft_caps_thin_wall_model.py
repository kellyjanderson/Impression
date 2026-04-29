"""Loft caps with thin-wall profile."""

from __future__ import annotations

from impression.modeling import loft
from impression.modeling.drawing2d import make_rect, Profile2D


def build():
    outer = make_rect(size=(2.0, 0.6)).outer
    inner = make_rect(size=(1.7, 0.3)).outer
    profile = Profile2D(outer=outer, holes=[inner])
    return loft(
        [profile, profile],
        start_cap="dome",
        end_cap="dome",
        start_cap_length=0.4,
        end_cap_length=0.4,
        cap_scale_dims="both",
    )


if __name__ == "__main__":
    build()
