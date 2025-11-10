# Modeling — Drafting Helpers

Utilities for creating thin 2.5D geometry useful in annotations, callouts, or technical documentation.

```python
from impression.modeling.drafting import make_line, make_plane, make_arrow, make_dimension
```

## Lines

```python
line = make_line((0, 0, 0), (1, 0.2, 0.5), thickness=0.03, color="#2d7dff")
```

Creates a rectangular rod between two points. Thickness controls cross-section.

## Planes

```python
plane = make_plane(size=(2, 1), center=(0, 0, 0), normal=(0, 1, 0), color=(0.9, 0.9, 0.95, 0.6))
```

Generates a quad oriented by `normal`, useful for section markers.

## Arrows

```python
arrow = make_arrow((0, 0, 0), (0.8, 0.2, 0.3), shaft_diameter=0.03, color="#ffb703")
```

## Dimensions

```python
meshes = make_dimension((0, 0, 0), (1.5, 0, 0), offset=0.15, text="1.50", color="#ff5a36")
```

Returns `[arrow_mesh, text_mesh]` so you can merge or preview them individually.

### Examples

- `docs/examples/drafting/line_plane_example.py`
- `docs/examples/drafting/dimension_example.py`

Preview commands:

```bash
impression preview docs/examples/drafting/line_plane_example.py
impression preview docs/examples/drafting/dimension_example.py
```
