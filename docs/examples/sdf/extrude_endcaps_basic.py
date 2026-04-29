"""SDF extrude with rounded endcaps (basic rectangle)."""

from __future__ import annotations

from impression.modeling import extrude_sdf
from impression.modeling.drawing2d import make_rect


def build():
    profile = make_rect(size=(2.0, 1.0))
    return extrude_sdf(
        profile,
        height=2.0,
        cap_radius=0.35,
        grid_spacing=0.12,
    )


if __name__ == "__main__":
    build()
