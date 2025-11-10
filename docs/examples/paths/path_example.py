"""Path utility demo."""

from __future__ import annotations

from impression.modeling import Path

path = Path.from_points(
    [
        (0, 0, 0),
        (1, 0, 0),
        (2, 1, 0.5),
        (3, 1.5, 1.0),
    ],
    closed=False,
)

print("Length:", round(path.length(), 3))
print("Samples:\n", path.sample(6))
polyline = path.to_polyline()
print("Polyline cells:", polyline.n_cells)
