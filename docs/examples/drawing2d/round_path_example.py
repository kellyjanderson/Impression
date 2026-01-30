"""Rounded path example using true arcs.

Run with:
  impression preview docs/examples/drawing2d/round_path_example.py
"""

from __future__ import annotations

from impression.modeling.drawing2d import round_corners, Profile2D
from impression.modeling import linear_extrude


def build():
    pts = [(0, 0), (2, 0), (2, 1), (0, 1)]
    rounded_path = round_corners(pts, radius=0.2, closed=True)
    profile = Profile2D(outer=rounded_path)
    return linear_extrude(profile, height=0.2)


if __name__ == "__main__":
    build()
