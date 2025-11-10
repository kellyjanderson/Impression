# Modeling — Primitives

All primitives are exposed from `impression.modeling` and currently use PyVista meshes under the hood. Import helpers like:

```python
from impression.modeling import make_box, make_cylinder, make_sphere, make_torus
```

> Rendering note: the CLI preview (`impression preview`) or `docs/examples/...` scripts can be used to visualize outputs on a desktop environment.

## Box

- **Function:** `make_box(size=(dx, dy, dz), center=(0, 0, 0))`
- **Options**
  - `size`: tuple specifying side lengths along X/Y/Z.
  - `center`: world-space center of the box.
- **Example:** `docs/examples/primitives/box_example.py`
- **Preview:** `impression preview docs/examples/primitives/box_example.py`

## Cylinder

- **Function:** `make_cylinder(radius=0.5, height=1.0, center=(0,0,0), direction=(0,0,1), resolution=128)`
- **Options**
  - `radius` / `height`
  - `direction`: normalized axis vector.
  - `resolution`: number of segments around the circumference.
- **Example:** `docs/examples/primitives/cylinder_example.py`
- **Preview:** `impression preview docs/examples/primitives/cylinder_example.py`

## Sphere

- **Function:** `make_sphere(radius=0.5, center=(0,0,0), theta_resolution=64, phi_resolution=64)`
- **Options:** radius, center, longitudinal (`theta_resolution`) and latitudinal (`phi_resolution`) segment counts.
- **Example:** `docs/examples/primitives/sphere_example.py`
- **Preview:** `impression preview docs/examples/primitives/sphere_example.py`

## Torus

- **Function:** `make_torus(major_radius=1.0, minor_radius=0.25, center=(0,0,0), direction=(0,0,1), n_theta=64, n_phi=32)`
- **Options**
  - `major_radius`: distance from center to tube centerline.
  - `minor_radius`: tube radius.
  - `direction`: orientation axis.
  - `n_theta` / `n_phi`: angular resolutions.
- **Example:** `docs/examples/primitives/torus_example.py`
- **Preview:** `impression preview docs/examples/primitives/torus_example.py`
