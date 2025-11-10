from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

import numpy as np
import pyvista as pv

Backend = Literal["mesh"]


def _ensure_backend(backend: Backend) -> None:
    if backend != "mesh":
        raise ValueError(f"Unsupported backend '{backend}'. Only 'mesh' is available right now.")


def make_box(
    size: Sequence[float] = (1.0, 1.0, 1.0),
    center: Sequence[float] = (0.0, 0.0, 0.0),
    backend: Backend = "mesh",
) -> pv.PolyData:
    """Axis-aligned box specified by size (dx, dy, dz) and center."""

    _ensure_backend(backend)
    sx, sy, sz = size
    cx, cy, cz = center
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    bounds = (cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz)
    return pv.Box(bounds=bounds)


def make_cylinder(
    radius: float = 0.5,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    resolution: int = 128,
    backend: Backend = "mesh",
) -> pv.PolyData:
    """Right circular cylinder aligned with `direction`."""

    _ensure_backend(backend)
    direction = _normalize(direction)
    mesh = pv.Cylinder(
        center=center,
        direction=direction,
        radius=radius,
        height=height,
        resolution=resolution,
    )
    return mesh.triangulate()


def make_sphere(
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    theta_resolution: int = 64,
    phi_resolution: int = 64,
    backend: Backend = "mesh",
) -> pv.PolyData:
    _ensure_backend(backend)
    return pv.Sphere(
        radius=radius,
        center=center,
        theta_resolution=theta_resolution,
        phi_resolution=phi_resolution,
    )


def make_torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.25,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    n_theta: int = 64,
    n_phi: int = 32,
    backend: Backend = "mesh",
) -> pv.PolyData:
    """Generate a torus (donut) with given major/minor radii."""

    _ensure_backend(backend)
    direction = _normalize(direction)
    base = pv.ParametricTorus(
        ringradius=major_radius,
        crosssectionradius=minor_radius,
        u_res=n_theta,
        v_res=n_phi,
    ).triangulate()

    aligned = _orient_mesh(base, direction)
    aligned.translate(center, inplace=True)
    return aligned


def make_cone(
    bottom_diameter: float = 1.0,
    top_diameter: float = 0.0,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    resolution: int = 64,
    backend: Backend = "mesh",
) -> pv.PolyData:
    """Circular frustum. Set top_diameter=0 for a classic cone."""

    _ensure_backend(backend)
    bottom_radius = bottom_diameter / 2.0
    top_radius = top_diameter / 2.0
    if bottom_radius <= 0 and top_radius <= 0:
        raise ValueError("At least one of bottom_diameter or top_diameter must be > 0.")

    mesh = _circular_frustum_mesh(bottom_radius, top_radius, height, resolution)
    mesh = _orient_mesh(mesh, direction)
    mesh.translate(center, inplace=True)
    return mesh


def make_prism(
    base_size: Sequence[float] = (1.0, 1.0),
    top_size: Sequence[float] | None = None,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    backend: Backend = "mesh",
) -> pv.PolyData:
    """
    Rectangular frustum (pyramid/prism). Set top_size=(0,0) for a pyramid, or None to match base.
    """

    _ensure_backend(backend)
    if top_size is None:
        top_size = tuple(base_size)
    mesh = _rectangular_frustum_mesh(tuple(base_size), tuple(top_size), height)
    mesh = _orient_mesh(mesh, direction)
    mesh.translate(center, inplace=True)
    return mesh


def _normalize(vector: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Direction vector must be non-zero.")
    arr = arr / norm
    return float(arr[0]), float(arr[1]), float(arr[2])


def _orient_mesh(mesh: pv.PolyData, direction: Sequence[float]) -> pv.PolyData:
    target = np.asarray(direction, dtype=float)
    target_norm = np.linalg.norm(target)
    if target_norm == 0:
        raise ValueError("Direction vector must be non-zero.")
    target = target / target_norm
    default = np.array([0.0, 0.0, 1.0])
    if np.allclose(target, default):
        return mesh.copy()
    axis = np.cross(default, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        # opposite direction; rotate 180 around X
        axis = np.array([1.0, 0.0, 0.0])
        angle_deg = 180.0
    else:
        axis = axis / axis_norm
        angle_rad = np.arccos(np.clip(np.dot(default, target), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
    rotated = mesh.copy()
    rotated.rotate_vector(axis, angle_deg, point=(0.0, 0.0, 0.0), inplace=True)
    return rotated


def _circular_frustum_mesh(
    bottom_radius: float,
    top_radius: float,
    height: float,
    resolution: int,
) -> pv.PolyData:
    bottom_radius = max(bottom_radius, 0.0)
    top_radius = max(top_radius, 0.0)
    z_bottom = -height / 2.0
    z_top = height / 2.0
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    points = []
    faces = []

    def ring_points(radius: float, z: float) -> np.ndarray:
        return np.column_stack(
            [
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.full_like(angles, z),
            ]
        )

    bottom_has_ring = bottom_radius > 0
    top_has_ring = top_radius > 0

    if bottom_has_ring:
        bottom = ring_points(bottom_radius, z_bottom)
        bottom_indices = np.arange(len(bottom))
        points.append(bottom)
    else:
        bottom = np.array([[0.0, 0.0, z_bottom]])
        bottom_indices = np.array([0])
        points.append(bottom)

    if top_has_ring:
        top = ring_points(top_radius, z_top)
        top_indices = np.arange(len(points[0]), len(points[0]) + len(top))
        points.append(top)
    else:
        top = np.array([[0.0, 0.0, z_top]])
        top_indices = np.array([len(points[0])])
        points.append(top)

    points_arr = np.vstack(points)

    if bottom_has_ring and top_has_ring:
        count = resolution
        for i in range(count):
            j = (i + 1) % count
            faces.append([4, bottom_indices[i], bottom_indices[j], top_indices[j], top_indices[i]])
        faces.append([count, *bottom_indices])
        faces.append([count, *top_indices[::-1]])
    elif bottom_has_ring:
        apex = int(top_indices[0])
        count = resolution
        for i in range(count):
            j = (i + 1) % count
            faces.append([3, bottom_indices[i], bottom_indices[j], apex])
        faces.append([count, *bottom_indices])
    else:
        # inverted cone (top ring, bottom apex)
        apex = int(bottom_indices[0])
        count = resolution
        for i in range(count):
            j = (i + 1) % count
            faces.append([3, apex, top_indices[j], top_indices[i]])
        faces.append([count, *top_indices[::-1]])

    faces_arr = np.hstack(faces)
    return pv.PolyData(points_arr, faces_arr).clean()


def _rectangular_frustum_mesh(
    base_size: Tuple[float, float],
    top_size: Tuple[float, float],
    height: float,
) -> pv.PolyData:
    hx, hy = base_size[0] / 2.0, base_size[1] / 2.0
    tx, ty = top_size[0] / 2.0, top_size[1] / 2.0
    z_bottom = -height / 2.0
    z_top = height / 2.0

    bottom_pts = np.array(
        [
            (-hx, -hy, z_bottom),
            (hx, -hy, z_bottom),
            (hx, hy, z_bottom),
            (-hx, hy, z_bottom),
        ]
    )

    if top_size[0] == 0 and top_size[1] == 0:
        top_pts = np.array([[0.0, 0.0, z_top]])
        apex_only = True
    else:
        top_pts = np.array(
            [
                (-tx, -ty, z_top),
                (tx, -ty, z_top),
                (tx, ty, z_top),
                (-tx, ty, z_top),
            ]
        )
        apex_only = False

    points = np.vstack([bottom_pts, top_pts])
    faces = []
    bottom_indices = np.arange(4)

    if apex_only:
        apex_idx = 4
        for i in range(4):
            j = (i + 1) % 4
            faces.append([3, bottom_indices[i], bottom_indices[j], apex_idx])
        faces.append([4, *bottom_indices])
    else:
        top_indices = np.arange(4, 8)
        for i in range(4):
            j = (i + 1) % 4
            faces.append([4, bottom_indices[i], bottom_indices[j], top_indices[j], top_indices[i]])
        faces.append([4, *bottom_indices])
        faces.append([4, *top_indices[::-1]])

    faces_arr = np.hstack(faces)
    return pv.PolyData(points, faces_arr).clean()
