"""SDF extrude with rounded endcaps (complex profile with holes)."""

from __future__ import annotations

from impression.modeling import extrude_sdf
from impression.modeling.drawing2d import make_rect, Profile2D


def build():
    outer = make_rect(size=(3.0, 1.8)).outer
    hole = make_rect(size=(1.2, 0.6)).outer
    profile = Profile2D(outer=outer, holes=[hole])
    return extrude_sdf(
        profile,
        height=2.0,
        cap_radius=0.3,
        grid_spacing=0.12,
    )


if __name__ == "__main__":
    build()
