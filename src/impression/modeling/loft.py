from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from impression.mesh import Mesh

from ._color import set_mesh_color
from ._profile2d import _loops_resampled, _triangulate_loops
from .drawing2d import Profile2D
from .path3d import Path3D
from .paths import Path as PolyPath


def loft_profiles(
    profiles: Sequence[Profile2D],
    path: Path3D | PolyPath | Sequence[Sequence[float]] | None = None,
    samples: int = 200,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
    cap_ends: bool = False,
) -> Mesh:
    """Loft a sequence of profiles, optionally along a path."""

    if len(profiles) < 2:
        raise ValueError("loft_profiles requires at least two profiles.")
    hole_count = len(profiles[0].holes)
    for profile in profiles[1:]:
        if len(profile.holes) != hole_count:
            raise ValueError("All profiles must have the same number of holes.")

    positions = _resolve_positions(path, len(profiles))

    loop_count = 1 + hole_count
    loops_per_profile = []
    for profile in profiles:
        loops = _loops_resampled(
            profile,
            samples,
            segments_per_circle=segments_per_circle,
            bezier_samples=bezier_samples,
        )
        loops_per_profile.append(loops)

    vertices = []
    offsets = []
    for profile_idx, loops in enumerate(loops_per_profile):
        profile_offsets = []
        for loop in loops:
            profile_offsets.append(len(vertices))
            pts3 = np.column_stack([loop, np.zeros(len(loop))]) + positions[profile_idx]
            vertices.extend(pts3)
        offsets.append(profile_offsets)

    vertices = np.asarray(vertices, dtype=float)

    faces = []
    for idx in range(len(profiles) - 1):
        for loop_idx in range(loop_count):
            start_a = offsets[idx][loop_idx]
            start_b = offsets[idx + 1][loop_idx]
            for i in range(samples):
                j = (i + 1) % samples
                a0 = start_a + i
                a1 = start_a + j
                b0 = start_b + i
                b1 = start_b + j
                faces.append([a0, a1, b1])
                faces.append([a0, b1, b0])

    if cap_ends:
        base_vertices, base_faces = _triangulate_loops(loops_per_profile[0])
        if base_faces.size:
            start_offset = offsets[0][0]
            end_offset = offsets[-1][0]
            faces.extend((base_faces + start_offset).tolist())
            faces.extend((base_faces[:, [0, 2, 1]] + end_offset).tolist())

    mesh = Mesh(vertices, np.asarray(faces, dtype=int))
    color = profiles[0].color
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def loft(
    profiles: Sequence[Profile2D],
    path: Path3D | PolyPath | Sequence[Sequence[float]] | None = None,
    samples: int = 200,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
    cap_ends: bool = False,
) -> Mesh:
    """Alias for loft_profiles."""

    return loft_profiles(
        profiles=profiles,
        path=path,
        samples=samples,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
        cap_ends=cap_ends,
    )


def _resolve_positions(
    path: Path3D | PolyPath | Sequence[Sequence[float]] | None,
    count: int,
) -> np.ndarray:
    if path is None:
        return np.column_stack([np.zeros(count), np.zeros(count), np.linspace(0.0, 1.0, count)])

    if isinstance(path, Path3D):
        pts = path.sample()
    elif isinstance(path, PolyPath):
        pts = np.asarray(path._effective_points(), dtype=float)
    else:
        pts = np.asarray(path, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("path must be a sequence of 3D points.")

    return _resample_path(pts, count)


def _resample_path(points: np.ndarray, count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("path sample count must be >= 2.")
    pts = np.asarray(points, dtype=float)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if pts.shape[0] == count:
        return pts.copy()
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg_lengths.sum())
    if total == 0:
        return np.tile(pts[0], (count, 1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    targets = np.linspace(0.0, total, count)
    result = []
    seg_index = 0
    for t in targets:
        while seg_index < len(seg_lengths) - 1 and cumulative[seg_index + 1] < t:
            seg_index += 1
        seg_start = cumulative[seg_index]
        seg_end = cumulative[seg_index + 1]
        p0 = pts[seg_index]
        p1 = pts[seg_index + 1]
        if seg_end == seg_start:
            result.append(p0)
        else:
            alpha = (t - seg_start) / (seg_end - seg_start)
            result.append((1 - alpha) * p0 + alpha * p1)
    return np.asarray(result, dtype=float)


__all__ = ["loft_profiles", "loft"]
