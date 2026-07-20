"""Caps that scale only the limiting dimension."""

from __future__ import annotations

from impression.modeling import loft
from impression.modeling.drawing2d import make_rect


def build():
    profiles = [
        make_rect(size=(2.0, 0.4)),
        make_rect(size=(1.4, 0.6)),
    ]
    return loft(
        profiles,
        start_cap="dome",
        end_cap="dome",
        start_cap_length=0.6,
        end_cap_length=0.6,
        cap_scale_dims="smallest",
    )


if __name__ == "__main__":
    build()
