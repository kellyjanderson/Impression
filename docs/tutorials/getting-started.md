# Tutorial - Getting Started

This tutorial walks through the basics: creating your first model, previewing it, and
exporting a watertight STL. Every modeling step uses surfaced Impression
primitives; tessellation occurs only in preview and export consumers.

## 1) Create a Simple Model

Create a new file `examples/hello_impression.py`:

```python
from impression.modeling import make_box, make_cylinder, boolean_union


def build():
    base = make_box(size=(40, 30, 6))
    post = make_cylinder(radius=5, height=18).translate((12, 8, 6))
    result = boolean_union([base, post])
    if result.status != "succeeded" or result.body is None:
        raise RuntimeError(result.failure_reason or "Surface union failed.")
    return result.body
```

Key ideas:

- `build()` returns a `SurfaceBody`, not a mesh or PyVista object.
- All dimensions use your configured units (default: millimeters).
- Public CSG accepts only `SurfaceBody` operands and returns a structured
  `SurfaceBooleanResult`; preview and export consume the accepted `result.body`.

## 2) Preview

```bash
impression preview examples/hello_impression.py
```

The preview window supports orbit, pan, and zoom. The file is watched by default, so
saving the model or an imported project-local module promptly reloads the preview.
Press `r` to force a cache-invalidating rebuild. If a rebuild fails, the last good
scene remains visible while the error is reported.

## 3) Export to STL

```bash
impression export examples/hello_impression.py --output dist/hello_impression.stl --overwrite
```

## 4) Add Color

Color metadata is retained on surfaced geometry and used in the viewer. This
does not affect STL export, but it helps visualize assemblies and CSG behavior.

```python
from impression.modeling import make_box


def build():
    return make_box(size=(10, 10, 10)).with_color("#6ab0ff")
```

## 5) Next Steps

- Learn the modeling toolkit in [`docs/modeling/primitives.md`](../modeling/primitives.md) and
  [`docs/modeling/csg.md`](../modeling/csg.md).
- Try 2D profiles, text outlines, and loft workflows in
  [`docs/modeling/drawing2d.md`](../modeling/drawing2d.md),
  [`docs/modeling/text.md`](../modeling/text.md), and
  [`docs/modeling/loft.md`](../modeling/loft.md).
- Explore the example library under [`docs/examples/`](../examples/).
