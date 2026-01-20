# Modeling — Extrusions

Extrusions convert 2D profiles into 3D meshes. Profiles are defined with the 2D
drawing API and may include holes. Extrusion uses sampled curves and triangulates
profile caps with `mapbox_earcut`.

Dependency: `pip install mapbox_earcut`

Import helpers:

```python
from impression.modeling import linear_extrude, rotate_extrude
from impression.modeling.drawing2d import make_rect, make_circle
```

## Linear Extrude

- **Function:** `linear_extrude(profile, height=1.0, direction=(0,0,1))`
- **Returns:** `Mesh`

```python
from impression.modeling import linear_extrude
from impression.modeling.drawing2d import make_rect

def build():
    profile = make_rect(size=(2.0, 1.0))
    return linear_extrude(profile, height=1.5)
```

Example: `docs/examples/extrusions/linear_extrude_example.py`

## Rotate Extrude

- **Function:** `rotate_extrude(profile, angle_deg=360, axis_origin=(0,0,0), axis_direction=(0,0,1))`
- **Returns:** `Mesh`
- **Notes:** profile is interpreted in a plane containing the axis; set `plane_normal`
  to control orientation (default `(0,1,0)` gives an XZ profile for a Z axis).

```python
from impression.modeling import rotate_extrude
from impression.modeling.drawing2d import make_polygon

def build():
    profile = make_polygon([(0.4, -0.6), (0.8, 0.0), (0.4, 0.6), (0.2, 0.2)])
    return rotate_extrude(profile, angle_deg=360)
```

Example: `docs/examples/extrusions/rotate_extrude_example.py`
