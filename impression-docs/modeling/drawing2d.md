# Modeling — 2D Drawing

The 2D drawing API provides profile- and path-based primitives that keep true curve
segments (lines, arcs, beziers). Open shapes return `Path2D`. Filled-shape authoring
is available through `PlanarShape2D`, and kernel modeling consumes topology-native
`Region`/`Section` via `as_section`.

Import helpers from the 2D module:

```python
from impression.modeling.drawing2d import make_rect, make_circle, make_ngon, make_polyline
```

Example module: `docs/examples/drawing2d/basic_example.py`

## Ownership Matrix

`drawing2d` is the authored-geometry surface. Topology ownership lives in `topology`.

Owned by `drawing2d`:

- authored segments and paths: `Line2D`, `Arc2D`, `Bezier2D`, `Path2D`
- user-facing 2D constructors: `make_rect`, `make_circle`, `make_ngon`, `make_polygon`, `make_polyline`
- curve authoring helpers: `round_corners`, `round_path`

Not owned by `drawing2d`:

- winding/classification policy for planar solids
- triangulation kernels
- region/section correspondence and matching
- hole/region topology event handling

See:

- [Topology](topology.md) for loop/region/section ownership and kernel helpers
- [Topology Spec 05](../../project/specifications/topology-05-drawing2d-boundary.md) for the boundary contract

### Example Catalog

- `docs/examples/drawing2d/line2d_example.py`
- `docs/examples/drawing2d/arc2d_example.py`
- `docs/examples/drawing2d/bezier2d_example.py`
- `docs/examples/drawing2d/path2d_example.py`
- `docs/examples/drawing2d/planarshape2d_example.py`
- `docs/examples/drawing2d/rect2d_example.py`
- `docs/examples/drawing2d/circle2d_example.py`
- `docs/examples/drawing2d/ngon2d_example.py`
- `docs/examples/drawing2d/polygon2d_example.py`
- `docs/examples/drawing2d/polyline2d_example.py`
- `docs/examples/drawing2d/round_path_example.py`

## Core Types

- **Path2D**: ordered segments (line, arc, bezier), open or closed.
- **PlanarShape2D**: filled-shape container with an outer boundary
  and optional holes. Kernel planar-topology operations should use
  `Loop`/`Region`/`Section` from topology.

### Open vs Closed Paths

`Path2D` can be open (`closed=False`) or closed (`closed=True`). Open paths are
useful for strokes and construction geometry; closed paths are used to define
filled profiles.

```python
from impression.modeling.drawing2d import make_polyline

open_path = make_polyline([(0, 0), (1, 0.5), (2, 0)], closed=False)
closed_path = make_polyline([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True)
```

### Holes and Winding

Profiles support holes via a list of closed `Path2D` loops. Winding is preserved
but not enforced yet. `PlanarShape2D` assumes closed loops; validation will come later.
For future boolean/extrude operations, follow this rule:

- Outer boundary: counter-clockwise (CCW)
- Holes: clockwise (CW)

Arcs expose a `clockwise` flag to control winding direction explicitly.

Both support per-object color. Profiles with holes can be created directly from
paths:

```python
from impression.modeling.drawing2d import Path2D, PlanarShape2D, Arc2D

outer = Path2D([Arc2D(center=(0, 0), radius=1.0, start_angle_deg=0, end_angle_deg=360)], closed=True)
inner = Path2D([Arc2D(center=(0, 0), radius=0.4, start_angle_deg=0, end_angle_deg=360, clockwise=True)], closed=True)
ring = PlanarShape2D(outer=outer, holes=[inner])
```

## Rect

- **Function:** `make_rect(size=(w, h), center=(x, y))`
- **Returns:** `PlanarShape2D`

```python
from impression.modeling.drawing2d import make_rect

def build():
    return make_rect(size=(2.0, 1.0), center=(0.0, 0.0))
```

## Circle

- **Function:** `make_circle(radius=0.5, center=(x, y))`
- **Returns:** `PlanarShape2D`

```python
from impression.modeling.drawing2d import make_circle

def build():
    return make_circle(radius=0.8, center=(0.0, 0.0))
```

## N-gon

- **Function:** `make_ngon(sides=6, radius=0.5, center=(x, y))`
- **Returns:** `PlanarShape2D`

```python
from impression.modeling.drawing2d import make_ngon

def build():
    return make_ngon(sides=5, radius=0.9)
```

## Polyline

- **Function:** `make_polyline(points, closed=False)`
- **Returns:** `Path2D`

```python
from impression.modeling.drawing2d import make_polyline

def build():
    return make_polyline([(0, 0), (1, 0.5), (2, 0)], closed=False)
```

## Rounded Corners

Use true arcs to soften sharp corners on a polyline or path.

- **Function:** `round_corners(points, radius, closed=True)`
- **Function:** `round_path(path, radius, clamp=True)`
- **Returns:** `Path2D` (with arc segments replacing sharp corners)

```python
from impression.modeling.drawing2d import round_corners

def build():
    pts = [(0, 0), (2, 0), (2, 1), (0, 1)]
    return round_corners(pts, radius=0.2, closed=True)
```
