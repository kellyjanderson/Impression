from __future__ import annotations

from typing import Iterable

import numpy as np
from impression.modeling.drawing2d import Path2D, Profile2D


def _loop_points(
    path: Path2D,
    segments_per_circle: int,
    bezier_samples: int,
) -> np.ndarray:
    pts = path.sample(segments_per_circle=segments_per_circle, bezier_samples=bezier_samples)
    if pts.shape[0] == 0:
        return pts
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def _signed_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_winding(points: np.ndarray, clockwise: bool) -> np.ndarray:
    if points.shape[0] < 3:
        return points
    area = _signed_area(points)
    is_cw = area < 0
    if is_cw != clockwise:
        return points[::-1].copy()
    return points


def _profile_loops(
    profile: Profile2D,
    segments_per_circle: int,
    bezier_samples: int,
    enforce_winding: bool = True,
) -> list[np.ndarray]:
    loops: list[np.ndarray] = []
    outer = _loop_points(profile.outer, segments_per_circle, bezier_samples)
    if enforce_winding:
        outer = _ensure_winding(outer, clockwise=False)
    loops.append(outer)
    for hole in profile.holes:
        pts = _loop_points(hole, segments_per_circle, bezier_samples)
        if enforce_winding:
            pts = _ensure_winding(pts, clockwise=True)
        loops.append(pts)
    return loops


def _triangulate_profile(
    profile: Profile2D,
    segments_per_circle: int,
    bezier_samples: int,
    enforce_winding: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    try:
        import mapbox_earcut as earcut
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("mapbox_earcut is required for profile triangulation.") from exc

    loops = _profile_loops(profile, segments_per_circle, bezier_samples, enforce_winding)
    if not loops:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=int), loops

    vertices = np.vstack(loops).astype(np.float32)
    ring_ends = []
    offset = 0
    for loop in loops:
        offset += loop.shape[0]
        ring_ends.append(offset)
    ring_end_indices = np.asarray(ring_ends, dtype=np.uint32)
    indices = earcut.triangulate_float32(vertices, ring_end_indices)
    faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    return vertices.astype(float), faces, loops


def _triangulate_loops(loops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not loops:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=int)
    try:
        import mapbox_earcut as earcut
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("mapbox_earcut is required for profile triangulation.") from exc

    vertices = np.vstack(loops).astype(np.float32)
    ring_ends = []
    offset = 0
    for loop in loops:
        offset += loop.shape[0]
        ring_ends.append(offset)
    ring_end_indices = np.asarray(ring_ends, dtype=np.uint32)
    indices = earcut.triangulate_float32(vertices, ring_end_indices)
    faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    return vertices.astype(float), faces


def _resample_loop(points: np.ndarray, count: int) -> np.ndarray:
    if count < 3:
        raise ValueError("resample count must be >= 3.")
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("Loop requires at least two points.")
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if pts.shape[0] == count:
        return pts.copy()

    closed = np.vstack([pts, pts[0]])
    seg_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg_lengths.sum())
    if total == 0:
        return np.tile(pts[0], (count, 1))

    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    targets = np.linspace(0.0, total, count + 1)[:-1]
    result = []
    seg_index = 0
    for t in targets:
        while seg_index < len(seg_lengths) - 1 and cumulative[seg_index + 1] < t:
            seg_index += 1
        seg_start = cumulative[seg_index]
        seg_end = cumulative[seg_index + 1]
        p0 = closed[seg_index]
        p1 = closed[seg_index + 1]
        if seg_end == seg_start:
            result.append(p0)
        else:
            alpha = (t - seg_start) / (seg_end - seg_start)
            result.append((1 - alpha) * p0 + alpha * p1)
    return np.asarray(result, dtype=float)


def _loops_resampled(
    profile: Profile2D,
    count: int,
    segments_per_circle: int,
    bezier_samples: int,
    enforce_winding: bool = True,
) -> list[np.ndarray]:
    loops = _profile_loops(profile, segments_per_circle, bezier_samples, enforce_winding)
    return [_resample_loop(loop, count) for loop in loops]
