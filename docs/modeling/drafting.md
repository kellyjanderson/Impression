# Modeling — Drafting Helpers

Utilities for creating thin 2.5D geometry useful in annotations, callouts, or technical documentation.

```python
from impression.modeling.drafting import make_line, make_plane, make_arrow, make_dimension
```

The drafting helpers now support a surface-first path via `backend="surface"`.
That path returns canonical `SurfaceBody` outputs for line/plane/arrow and a
`SurfaceConsumerCollection` for multi-part dimensions, with tessellation only at
preview/export boundaries.

## Lines

```python
line = make_line((0, 0, 0), (1, 0.2, 0.5), thickness=0.03, color="#2d7dff")
surface_line = make_line((0, 0, 0), (1, 0.2, 0.5), thickness=0.03, color="#2d7dff", backend="surface")
```

Creates a rectangular rod between two points. Thickness controls cross-section.

## Planes

```python
plane = make_plane(size=(2, 1), center=(0, 0, 0), normal=(0, 1, 0), color=(0.9, 0.9, 0.95, 0.6))
surface_plane = make_plane(size=(2, 1), center=(0, 0, 0), normal=(0, 1, 0), color=(0.9, 0.9, 0.95, 0.6), backend="surface")
```

Generates a quad oriented by `normal`, useful for section markers.

## Arrows

```python
arrow = make_arrow((0, 0, 0), (0.8, 0.2, 0.3), shaft_diameter=0.03, color="#ffb703")
surface_arrow = make_arrow((0, 0, 0), (0.8, 0.2, 0.3), shaft_diameter=0.03, color="#ffb703", backend="surface")
```

## Dimensions

```python
meshes = make_dimension((0, 0, 0), (1.5, 0, 0), offset=0.15, text="1.50", color="#ff5a36")
surface_dimension = make_dimension((0, 0, 0), (1.5, 0, 0), offset=0.15, text="1.50", color="#ff5a36", backend="surface")
```

Returns a list containing the dimension arrow mesh and, when `text` is set, a text label mesh.
On the surface path, returns a `SurfaceConsumerCollection` preserving the arrow
body and optional label body as surfaced items.

Optional label controls:

- `font`: font family name lookup (default: `"Arial"`).
- `font_path`: explicit font file path (recommended for deterministic output).

If the selected font cannot be resolved, `make_dimension` keeps the arrow and skips label geometry.

### Examples

- `docs/examples/drafting/line_plane_example.py`
- `docs/examples/drafting/dimension_example.py`

Preview commands:

```bash
impression preview docs/examples/drafting/line_plane_example.py
impression preview docs/examples/drafting/dimension_example.py
```
