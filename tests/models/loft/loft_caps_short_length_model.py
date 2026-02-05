"""Caps with very short cap length."""

from __future__ import annotations

from impression.modeling import loft
from impression.modeling.drawing2d import make_rect


def build():
    profiles = [
        make_rect(size=(1.2, 0.8)),
        make_rect(size=(0.9, 0.6)),
    ]
    return loft(
        profiles,
        start_cap="slope",
        end_cap="slope",
        start_cap_length=0.15,
        end_cap_length=0.15,
        cap_scale_dims="both",
    )


if __name__ == "__main__":
    build()
