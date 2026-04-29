"""SDF loft with rounded endcaps (basic rectangles)."""

from __future__ import annotations

from impression.modeling import loft_sdf
from impression.modeling.drawing2d import make_rect


def build():
    profiles = [
        make_rect(size=(2.0, 1.0)),
        make_rect(size=(1.4, 1.6)),
    ]
    return loft_sdf(
        profiles,
        positions=[0.0, 1.5],
        cap_radius=0.35,
        grid_spacing=0.15,
    )


if __name__ == "__main__":
    build()
