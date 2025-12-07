# Transforms: translate & rotate

Impression includes lightweight helpers to move and orient meshes after you create them.

## API

- `translate(mesh, offset)` — returns a copy shifted by `(dx, dy, dz)`.
- `rotate(mesh, axis, angle_deg, origin=(0,0,0))` — returns a copy rotated `angle_deg` degrees about an arbitrary axis that passes through `origin`.

Both helpers mutate the mesh you pass in and return it; call .copy() first if you need to preserve an original.

## Example

```python
from impression.modeling import make_box, translate, rotate


def build():
    base = make_box(size=(0.8, 0.8, 0.4), center=(0.0, 0.0, 0.2), color="#5A7BFF")
    shifted = translate(base.copy(), (1.2, 0.0, 0.0))
    turned = rotate(base.copy(), axis=(0.0, 0.0, 1.0), angle_deg=45.0)
    return [base, shifted, turned]
```

Run it directly:

```bash
impression preview docs/examples/transforms/transform_example.py
```

The example will render three boxes: the base at the origin, a translated copy to the +X side, and a rotated copy about +Z.
