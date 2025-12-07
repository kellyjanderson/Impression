# Modeling — CSG Helpers

Imported from `impression.modeling`:

```python
from impression.modeling import boolean_union, boolean_difference, boolean_intersection, union_meshes
```

Booleans are color-aware: each source mesh can carry per-object color and the resulting faces inherit the color of the geometry they originated from (difference faces pick up the cutter color, intersections respect the nearest source, etc.).

All helpers currently operate on PyVista `PolyData` meshes. Inputs are triangulated/cleaned automatically.

## boolean_union(meshes, tolerance=1e-5)
- Combine two or more meshes into a single body.
- `meshes`: iterable of `pv.DataSet`.
- `tolerance`: geometric tolerance passed to PyVista booleans.
- Example: `docs/examples/csg/union_example.py` (blue box + orange cylinder demonstrates color inheritance)
- Preview: `impression preview docs/examples/csg/union_example.py`

```python
from impression.modeling import boolean_union, make_box, make_cylinder

def build():
    box = make_box(size=(2, 2, 1))
    cyl = make_cylinder(radius=0.6, height=1.5)
    return boolean_union([box, cyl])
```

![Union CSG](../assets/previews/csg-union.png)

You can also union a collection directly with `union_meshes`, which accepts either an iterable or a mapping (e.g., dict) of meshes:

```python
from impression.modeling import make_box, make_cylinder, union_meshes

def build():
    a = make_box(size=(2, 2, 1), color="#5A7BFF")
    b = make_cylinder(radius=0.8, height=1.5, color="#FF7A18")
    return union_meshes({"box": a, "cyl": b})
```

Example: `docs/examples/csg/union_meshes_example.py`

## boolean_difference(base, cutters, tolerance=1e-5)
- Subtract one or more cutter meshes from `base`.
- Example: `docs/examples/csg/difference_example.py` (cut plane adopts cutter color)
- Preview: `impression preview docs/examples/csg/difference_example.py`

```python
from impression.modeling import boolean_difference, make_box, make_cylinder

def build():
    base = make_box(size=(2, 2, 2))
    cutter = make_cylinder(radius=0.4, height=2.5)
    return boolean_difference(base, [cutter])
```

![Difference CSG](../assets/previews/csg-difference.png)

## boolean_intersection(meshes, tolerance=1e-5)
- Keep only overlapping volume among provided meshes.
- Example: `docs/examples/csg/intersection_example.py`
- Preview: `impression preview docs/examples/csg/intersection_example.py`

```python
from impression.modeling import boolean_intersection, make_box, make_sphere

def build():
    box = make_box(size=(2, 2, 2))
    sphere = make_sphere(radius=1.2)
    return boolean_intersection([box, sphere])
```

![Intersection CSG](../assets/previews/csg-intersection.png)
