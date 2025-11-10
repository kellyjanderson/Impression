# Modeling — CSG Helpers

Imported from `impression.modeling`:

```python
from impression.modeling import boolean_union, boolean_difference, boolean_intersection
```

All helpers currently operate on PyVista `PolyData` meshes. Inputs are triangulated/cleaned automatically.

## boolean_union(meshes, tolerance=1e-5)
- Combine two or more meshes into a single body.
- `meshes`: iterable of `pv.DataSet`.
- `tolerance`: geometric tolerance passed to PyVista booleans.
- Example: `docs/examples/csg/union_example.py`

## boolean_difference(base, cutters, tolerance=1e-5)
- Subtract one or more cutter meshes from `base`.
- Example: `docs/examples/csg/difference_example.py`

## boolean_intersection(meshes, tolerance=1e-5)
- Keep only overlapping volume among provided meshes.
- Example: `docs/examples/csg/intersection_example.py`
