# Modeling - Heightmaps

Heightmaps let you generate meshes from images and displace existing meshes using image data.
Impression accepts any image format supported by Pillow (PNG, JPG, TIFF, etc), auto‑grayscales
color images, and treats `alpha == 0` as masked.

```python
from impression.modeling import heightmap, displace_heightmap
```

## heightmap

Create a heightfield mesh from an image.

```python
terrain = heightmap(
    "assets/terrain.png",
    height=8.0,
    xy_scale=0.2,
    center=(0, 0, 0),
    alpha_mode="mask",  # drop faces where alpha == 0
)
```

Options:

- `image`: path, PIL image, or numpy array (`HxW`, `HxWx3`, `HxWx4`).
- `height`: vertical scale applied to grayscale values.
- `xy_scale`: scalar or `(sx, sy)` spacing between pixels.
- `center`: world‑space center of the heightfield.
- `alpha_mode`: `"mask"` (holes) or `"ignore"` (transparent pixels become zero height).

## displace_heightmap

Displace an existing mesh by projecting a heightmap onto it.

```python
from impression.modeling import make_box

box = make_box(size=(2, 2, 2))
carved = displace_heightmap(
    box,
    "assets/logo.png",
    height=0.3,
    plane="xy",
    direction="normal",
    alpha_mode="ignore",
)
```

Options:

- `projection`: currently only `"planar"` is supported.
- `plane`: `"xy"`, `"xz"`, or `"yz"` for planar projection.
- `direction`: `"normal"`, `"x"`, `"y"`, `"z"`, or a custom vector.
- `alpha_mode`: `"ignore"` (no displacement where alpha == 0) or `"mask"` (drop faces).
- `bounds`: optional `(umin, umax, vmin, vmax)` to override projection bounds.

## Notes

- Transparency is treated as a mask when `alpha_mode="mask"`.
- For basic stamping onto objects, planar projection + `direction="normal"` works well.
- UV‑based or triplanar projection is not implemented yet.

## Examples

- `docs/examples/heightmaps/heightmap_basic.py`
