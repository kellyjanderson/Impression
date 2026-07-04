# Modeling — Path3D

Path3D mirrors the 2D path API but stores true 3D curve segments (lines, arcs,
beziers). It is intended for sweeps/lofts and future tooling.

```python
from impression.modeling import Path3D, Line3D, Arc3D, Bezier3D
```

## Path3D

- `Path3D(segments, closed=False)`
- `Path3D.from_points(points, closed=False)` for straight segments.
- `Path3D.sample()` converts segments to sampled points for preview/export.

```python
from impression.modeling import Path3D, Line3D, Arc3D

segments = [
    Line3D(start=(-1, 0, 0), end=(0, 0.5, 0)),
    Arc3D(center=(0.5, 0.0, 0.0), radius=0.7, start_angle_deg=180, end_angle_deg=20, normal=(0, 0, 1)),
]
path = Path3D(segments)
```

Example: `docs/examples/paths/path3d_example.py`
