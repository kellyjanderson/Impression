"""Loft endcap variants for edge-case testing."""

from __future__ import annotations

from impression.modeling import loft
from impression.modeling.drawing2d import make_rect


def build():
    profiles = [
        make_rect(size=(1.2, 0.8)),
        make_rect(size=(0.7, 1.1)),
    ]
    return loft(
        profiles,
        start_cap="dome",
        end_cap="taper",
        start_cap_length=0.6,
        end_cap_length=0.6,
        cap_scale_dims="both",
    )


if __name__ == "__main__":
    build()
