from __future__ import annotations

from impression.modeling import make_bistable_hinge


def build():
    return make_bistable_hinge(
        width=42.0,
        height=16.0,
        thickness=2.0,
        anchor_width=10.0,
        shuttle_width=8.0,
        shuttle_height=6.0,
        ligament_width=1.3,
        preload_offset=2.6,
    )
