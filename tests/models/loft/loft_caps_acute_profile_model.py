"""Caps on a sharp/acute polygon profile."""

from __future__ import annotations

from impression.modeling import loft
from impression.modeling.drawing2d import make_polygon


def build():
    sharp = make_polygon(
        [
            (0.0, -0.8),
            (1.2, 0.0),
            (0.0, 0.8),
            (-1.2, 0.0),
        ]
    )
    profiles = [sharp, sharp]
    return loft(
        profiles,
        start_cap="taper",
        end_cap="dome",
        start_cap_length=0.5,
        end_cap_length=0.5,
        cap_scale_dims="both",
    )


if __name__ == "__main__":
    build()
