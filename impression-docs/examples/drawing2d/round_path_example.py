"""Rounded path example using true arcs.

Run with:
  impression preview docs/examples/drawing2d/round_path_example.py
"""

from __future__ import annotations

from impression.modeling.drawing2d import PlanarShape2D, round_corners


def build():
    pts = [(0, 0), (2, 0), (2, 1), (0, 1)]
    rounded_path = round_corners(pts, radius=0.2, closed=True)
    return PlanarShape2D(outer=rounded_path)


if __name__ == "__main__":
    build()
