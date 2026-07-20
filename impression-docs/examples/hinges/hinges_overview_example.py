from __future__ import annotations

from impression.modeling import (
    make_bistable_hinge,
    make_living_hinge,
    make_traditional_hinge_pair,
    translate,
)


def build():
    traditional = make_traditional_hinge_pair(opened_angle_deg=35.0, width=28.0, knuckle_count=5)
    living = make_living_hinge(width=52.0, height=22.0)
    bistable = make_bistable_hinge(width=38.0)

    translate(traditional, (-50.0, 0.0, 0.0))
    translate(living, (0.0, 0.0, 0.0))
    translate(bistable, (45.0, 0.0, 0.0))

    return [traditional, living, bistable]
