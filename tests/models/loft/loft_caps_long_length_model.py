"""Caps with long cap length."""

from __future__ import annotations

from impression.modeling import loft
from impression.modeling.drawing2d import make_rect


def build():
    profiles = [
        make_rect(size=(1.4, 0.7)),
        make_rect(size=(1.0, 0.9)),
    ]
    return loft(
        profiles,
        start_cap="dome",
        end_cap="taper",
        start_cap_length=2.5,
        end_cap_length=2.5,
        cap_scale_dims="both",
    )


if __name__ == "__main__":
    build()
