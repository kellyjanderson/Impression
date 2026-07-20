"""Loft caps on a concave profile."""

from __future__ import annotations

from impression.modeling import loft
from impression.modeling.drawing2d import make_polygon


def build():
    concave = make_polygon(
        [
            (-1.0, -0.4),
            (1.0, -0.4),
            (1.0, 0.2),
            (0.3, 0.2),
            (0.3, 0.8),
            (-0.3, 0.8),
            (-0.3, 0.2),
            (-1.0, 0.2),
        ]
    )
    profiles = [concave, concave]
    return loft(
        profiles,
        start_cap="taper",
        end_cap="taper",
        start_cap_length=0.5,
        end_cap_length=0.5,
        cap_scale_dims="both",
    )


if __name__ == "__main__":
    build()
