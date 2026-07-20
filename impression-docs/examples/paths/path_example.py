"""Path utility demo."""

from __future__ import annotations

from impression.modeling import Path


def build():
    path = Path.from_points(
        [
            (0, 0, 0),
            (1, 0, 0),
            (2, 1, 0.5),
            (3, 1.5, 1.0),
        ],
        closed=False,
    )
    # Return a polyline so preview can render it.
    return path.to_polyline()


if __name__ == "__main__":
    polyline = build()
    print("Length:", round(Path.from_points(polyline.points).length(), 3))
    print("Points:", len(polyline.points))
