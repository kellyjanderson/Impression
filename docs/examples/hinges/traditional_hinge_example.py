from __future__ import annotations

from impression.modeling import make_traditional_hinge_pair


def build():
    return make_traditional_hinge_pair(
        width=32.0,
        leaf_depth=14.0,
        leaf_thickness=2.4,
        barrel_diameter=6.5,
        pin_diameter=3.1,
        knuckle_count=7,
        knuckle_clearance=0.6,
        barrel_gap=0.8,
        attachment_overlap=0.25,
        connector_width_scale=0.55,
        opened_angle_deg=45.0,
        include_pin=True,
    )
