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
        u_resolution=n_theta,
        v_resolution=n_phi,
    ).triangulate()

    aligned = _orient_mesh(base, direction)
    aligned.translate(center, inplace=True)
    return aligned


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
