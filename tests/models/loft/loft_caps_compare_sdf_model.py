"""Compare SDF-based loft caps."""

from __future__ import annotations

from impression.modeling import loft
from impression.modeling.drawing2d import make_rect


def build():
    profiles = [
        make_rect(size=(1.2, 0.7)),
        make_rect(size=(0.8, 1.0)),
    ]
    return loft(
        profiles,
        cap_type="sdf",
        cap_radius=0.25,
        cap_grid_spacing=0.12,
    )


if __name__ == "__main__":
    build()
