from __future__ import annotations

import sys
from pathlib import Path

from impression.modeling import Bezier3D, Path3D, loft
from impression.modeling.drawing2d import Path2D, PlanarShape2D

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _common import make_label


def _profile(points, color: str) -> PlanarShape2D:
    return PlanarShape2D(outer=Path2D.from_points(points, closed=True)).with_color(color)


def build():
    base = _profile(
        [(-6.0, -3.0), (6.0, -3.0), (7.0, 2.2), (5.5, 4.0), (-5.5, 4.0), (-7.0, 2.2)],
        "#b8a387",
    )
    palm = _profile(
        [(-8.0, -3.5), (8.5, -3.5), (9.0, 1.8), (6.0, 5.5), (-5.0, 6.5), (-9.0, 2.0)],
        "#b8a387",
    )
    neck = _profile(
        [(-4.0, -2.3), (5.4, -2.3), (5.7, 1.2), (3.9, 3.7), (-2.8, 3.9), (-4.8, 1.5)],
        "#b8a387",
    )

    path = Path3D(
        [
            Bezier3D(
                p0=(0.0, 0.0, 0.0),
                p1=(1.0, -1.0, 12.0),
                p2=(2.5, 1.4, 26.0),
                p3=(3.0, 0.0, 36.0),
            )
        ]
    )

    grip = loft(
        [base, palm, neck],
        path=path,
        samples=96,
        cap_ends=True,
        start_cap="flat",
        end_cap="dome",
        end_cap_length=3.0,
        cap_steps=10,
        cap_scale_dims="both",
    )
    label = make_label("ERGONOMIC HANDLE", center=(2.0, 10.0, 20.0), font_size=6.4)
    return [grip, label]
