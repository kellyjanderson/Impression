"""Drafting helpers: line + plane"""

from __future__ import annotations

from impression.modeling.drafting import make_line, make_plane


def build():
    plane = make_plane(size=(2.0, 1.0), center=(0, 0, 0), normal=(0, 1, 0), color=(0.9, 0.9, 0.95, 0.7))
    line = make_line((-0.8, 0.0, -0.4), (0.8, 0.0, 0.4), thickness=0.04, color="#2d7dff")
    return [plane, line]


if __name__ == "__main__":
    print(len(build()))
