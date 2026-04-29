from __future__ import annotations

from impression.modeling import make_living_hinge


def build():
    return make_living_hinge(
        width=60.0,
        height=24.0,
        thickness=1.8,
        hinge_band_width=16.0,
        slit_width=0.55,
        slit_pitch=1.6,
        edge_margin=1.5,
        bridge=1.2,
    )
