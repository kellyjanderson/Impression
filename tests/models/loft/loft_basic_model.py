"""Basic loft model for tests."""

from __future__ import annotations

from impression.modeling import loft
from impression.modeling.drawing2d import make_rect


def build():
    profiles = [
        make_rect(size=(1.0, 0.6)),
        make_rect(size=(0.8, 1.0)),
        make_rect(size=(0.6, 0.4)),
    ]
    return loft(profiles)


if __name__ == "__main__":
    build()
