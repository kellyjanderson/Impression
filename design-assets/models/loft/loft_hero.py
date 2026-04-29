"""Hero render for loft documentation (design-assets)."""

from __future__ import annotations

from impression.modeling import loft, Path3D
from impression.modeling.drawing2d import make_rect, make_circle


def build():
    profiles = [
        make_rect(size=(1.4, 0.6)),
        make_circle(radius=0.55),
        make_rect(size=(0.7, 1.2)),
    ]
    path = Path3D.from_points(
        [
            (0.0, 0.0, 0.0),
            (0.4, 0.0, 1.0),
            (0.8, 0.3, 2.0),
            (1.0, 0.6, 2.8),
        ]
    )
    return loft(profiles, path=path, start_cap="dome", end_cap="dome", start_cap_length=0.6, end_cap_length=0.6)


if __name__ == "__main__":
    build()
