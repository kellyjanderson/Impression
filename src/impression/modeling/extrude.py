from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from impression.mesh import Mesh
from impression.mesh_quality import MeshQuality, apply_lod

from ._color import set_mesh_color
from ._profile2d import _profile_loops, _triangulate_profile
from .drawing2d import Profile2D


def _normalize(vector: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Direction vector must be non-zero.")
    return arr / norm


def _cap_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    expected_normal: np.ndarray,
) -> np.ndarray:
    if faces.size == 0:
        return faces
    tri = faces.copy()
    v1 = vertices[tri[:, 1]] - vertices[tri[:, 0]]
    v2 = vertices[tri[:, 2]] - vertices[tri[:, 0]]
    normals = np.cross(v1, v2)
    dots = np.einsum("ij,j->i", normals, expected_normal)
    flip = dots < 0
    if np.any(flip):
        tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri


def linear_extrude(
    profile: Profile2D,
    height: float = 1.0,
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    center: Sequence[float] = (0.0, 0.0, 0.0),
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
    quality: MeshQuality | None = None,
) -> Mesh:
    """Extrude a 2D profile along a straight direction."""

    height = float(height)
    if height <= 0:
        raise ValueError("height must be positive.")
    direction_vec = _normalize(direction) * height
    center_vec = np.asarray(center, dtype=float).reshape(3)

    if quality is not None:
        quality = apply_lod(quality)
        segments_per_circle = _apply_quality_samples(segments_per_circle, quality)
        bezier_samples = _apply_quality_samples(bezier_samples, quality)

    vertices_2d, faces_2d, loops = _triangulate_profile(
        profile,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    )
    if vertices_2d.size == 0:
        return Mesh(np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int))

    base = np.column_stack([vertices_2d[:, 0], vertices_2d[:, 1], np.zeros(len(vertices_2d))])
    base = base + center_vec
    top = base + direction_vec
    vertices = np.vstack([base, top])

    faces = []
    # caps
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=float)
    bottom = _cap_faces(base, faces_2d, expected_normal=-plane_normal)
    top_faces = faces_2d + len(base)
    top_faces = _cap_faces(top, top_faces - len(base), expected_normal=plane_normal) + len(base)
    faces.append(bottom)
    faces.append(top_faces)

    # sides
    offset = 0
    direction_sign = float(np.dot(_normalize(direction), plane_normal))
    for loop in loops:
        count = loop.shape[0]
        if count < 2:
            offset += count
            continue
        for i in range(count):
            j = (i + 1) % count
            b0 = offset + i
            b1 = offset + j
            t0 = b0 + len(base)
            t1 = b1 + len(base)
            if direction_sign >= 0:
                faces.append(np.array([[b0, b1, t1], [b0, t1, t0]], dtype=int))
            else:
                faces.append(np.array([[b0, t1, b1], [b0, t0, t1]], dtype=int))
        offset += count

    mesh = Mesh(vertices, np.vstack(faces))
    if profile.color is not None:
        set_mesh_color(mesh, profile.color)
    return mesh


def rotate_extrude(
    profile: Profile2D,
    angle_deg: float = 360.0,
    axis_origin: Sequence[float] = (0.0, 0.0, 0.0),
    axis_direction: Sequence[float] = (0.0, 0.0, 1.0),
    plane_normal: Sequence[float] = (0.0, 1.0, 0.0),
    segments: int = 64,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
    cap_ends: bool = True,
    quality: MeshQuality | None = None,
) -> Mesh:
    """Rotate-extrude (lathe) a profile around an axis."""

    axis_origin = np.asarray(axis_origin, dtype=float).reshape(3)
    axis_dir = _normalize(axis_direction)
    plane_norm = _normalize(plane_normal)
    if abs(float(np.dot(axis_dir, plane_norm))) > 1e-6:
        raise ValueError("plane_normal must be perpendicular to axis_direction.")
    u = np.cross(plane_norm, axis_dir)
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        raise ValueError("plane_normal cannot be parallel to axis_direction.")
    u = u / u_norm

    if quality is not None:
        quality = apply_lod(quality)
        segments = _apply_quality_samples(segments, quality)
        segments_per_circle = _apply_quality_samples(segments_per_circle, quality)
        bezier_samples = _apply_quality_samples(bezier_samples, quality)

    vertices_2d, faces_2d, loops = _triangulate_profile(
        profile,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    )
    if vertices_2d.size == 0:
        return Mesh(np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int))

    base_points = axis_origin + np.outer(vertices_2d[:, 0], u) + np.outer(vertices_2d[:, 1], axis_dir)

    angle_rad = np.deg2rad(angle_deg)
    closed = np.isclose(abs(angle_deg), 360.0)
    steps = max(int(segments), 3)
    if closed:
        angles = np.linspace(0.0, angle_rad, steps, endpoint=False)
    else:
        angles = np.linspace(0.0, angle_rad, steps + 1, endpoint=True)

    vertices = []
    for angle in angles:
        vertices.append(_rotate_around_axis(base_points, axis_origin, axis_dir, angle))
    vertices = np.vstack(vertices)

    ring_size = base_points.shape[0]
    ring_count = len(angles)
    faces = []

    # side faces
    for ring in range(ring_count - (0 if closed else 1)):
        next_ring = (ring + 1) % ring_count
        ring_offset = ring * ring_size
        next_offset = next_ring * ring_size
        for loop in loops:
            count = loop.shape[0]
            for i in range(count):
                j = (i + 1) % count
                b0 = ring_offset + i
                b1 = ring_offset + j
                t0 = next_offset + i
                t1 = next_offset + j
                if angle_deg >= 0:
                    faces.append(np.array([[b0, b1, t1], [b0, t1, t0]], dtype=int))
                else:
                    faces.append(np.array([[b0, t1, b1], [b0, t0, t1]], dtype=int))
            ring_offset += count
            next_offset += count

    if not closed and cap_ends:
        start_offset = 0
        end_offset = (ring_count - 1) * ring_size
        end_normal = _rotate_vector(plane_norm, axis_dir, angle_rad)
        if angle_deg >= 0:
            start_normal = -plane_norm
            final_normal = end_normal
        else:
            start_normal = plane_norm
            final_normal = -end_normal
        start_cap = _cap_faces(vertices, faces_2d + start_offset, expected_normal=start_normal)
        end_cap = _cap_faces(vertices, faces_2d + end_offset, expected_normal=final_normal)
        faces.append(start_cap)
        faces.append(end_cap)

    mesh = Mesh(vertices, np.vstack(faces))
    if profile.color is not None:
        set_mesh_color(mesh, profile.color)
    return mesh


def _rotate_around_axis(
    points: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    axis = _normalize(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    p = points - origin
    cross = np.cross(axis, p)
    dot = np.dot(p, axis)
    rotated = p * cos_a + cross * sin_a + axis * dot[:, None] * (1 - cos_a)
    return rotated + origin


def _rotate_vector(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis)
    vec = np.asarray(vec, dtype=float).reshape(3)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    cross = np.cross(axis, vec)
    dot = np.dot(axis, vec)
    return vec * cos_a + cross * sin_a + axis * dot * (1 - cos_a)

def _apply_quality_samples(value: int, quality: MeshQuality) -> int:
    if quality.lod == "preview":
        return max(6, int(value * 0.5))
    return value


__all__ = ["linear_extrude", "rotate_extrude"]
