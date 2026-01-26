# Transforms

Impression includes lightweight helpers to move, orient, and reshape meshes after creation.
These map closely to OpenSCAD transform vocabulary while keeping the internal mesh representation.

## API

- `translate(mesh, offset)` - shift by `(dx, dy, dz)`
- `rotate(mesh, axis, angle_deg, origin=(0,0,0))` - rotate about an axis
- `rotate(mesh, [x,y,z])` - Euler rotation (degrees), default order X then Y then Z
- `scale(mesh, factors, origin=(0,0,0))` - scale by `(sx, sy, sz)`
- `resize(mesh, size, auto=False)` - resize to exact bounds; preserve aspect where `auto` is true
- `mirror(mesh, axis)` - mirror across a plane whose normal is `axis`
- `multmatrix(mesh, m)` - apply a 4x4 transform matrix

All helpers mutate the mesh/group you pass in and return it; call `.copy()` if you need the original.

## Example

```python
from impression.modeling import make_box, translate, rotate, scale, mirror


def build():
    base = make_box(size=(1.0, 1.0, 0.4), color="#5A7BFF")
    shifted = translate(base.copy(), (1.4, 0.0, 0.0))
    turned = rotate(base.copy(), axis=(0.0, 0.0, 1.0), angle_deg=45.0)
    scaled = scale(base.copy(), (1.0, 0.5, 1.5))
    flipped = mirror(base.copy(), (1.0, 0.0, 0.0))
    return [base, shifted, turned, scaled, flipped]
```

Run it directly:

```bash
impression preview docs/examples/transforms/transform_example.py
```

## 2D Offsets and Hulls

Impression uses manifold3d CrossSection for 2D offset and convex hull:

- `offset(profile, r=..., delta=..., chamfer=False)`
- `hull([profile_a, profile_b, ...])`

These return one or more `Profile2D` instances.

## 3D Hulls

`hull([mesh_a, mesh_b, ...])` uses manifold3d to compute convex hulls for meshes.

## Minkowski

`minkowski(...)` is not yet supported with the current backend. It will require an
additional geometry kernel (e.g., CGAL/libigl) to implement 3D Minkowski sums.
