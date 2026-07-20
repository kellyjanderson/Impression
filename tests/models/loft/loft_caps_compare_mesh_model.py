"""Compare mesh-based loft caps."""

from __future__ import annotations

from impression.modeling import loft
from impression.modeling.drawing2d import make_rect


def build():
    profiles = [
        make_rect(size=(1.2, 0.7)),
        make_rect(size=(0.8, 1.0)),
    ]
    return loft(
        profiles,
        start_cap="dome",
        end_cap="dome",
        start_cap_length=0.6,
        end_cap_length=0.6,
        cap_scale_dims="both",
    )


if __name__ == "__main__":
    build()
