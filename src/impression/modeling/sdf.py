from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from impression.mesh import Mesh

from ._profile2d import _profile_loops
from .drawing2d import Profile2D
from .path3d import Path3D
from .paths import Path as PolyPath


@dataclass
class SdfGrid:
    values: np.ndarray
    origin: np.ndarray
    spacing: float


def extrude_sdf(
    profile: Profile2D,
    height: float = 1.0,
    cap_radius: float = 0.0,
    grid_spacing: float = 0.2,
    padding: float | None = None,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> Mesh:
    """Extrude a profile and apply SDF-based rounded endcaps."""

    height = float(height)
    if height <= 0:
        raise ValueError("height must be positive.")
    cap_radius = float(max(cap_radius, 0.0))
    grid_spacing = float(grid_spacing)
    if grid_spacing <= 0:
        raise ValueError("grid_spacing must be positive.")

    loops = _profile_loops(profile, segments_per_circle, bezier_samples, enforce_winding=True)
    if not loops or loops[0].size == 0:
        return Mesh(np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int))

    bbox_min, bbox_max = _loops_bbox(loops)
    base_pad = cap_radius if padding is None else float(padding)
    pad = base_pad + 3.0 * grid_spacing
    bbox_min = bbox_min - pad
    bbox_max = bbox_max + pad
    z_min = -height / 2.0 - pad
    z_max = height / 2.0 + pad

    grid = _sdf_grid_from_loops(
        loops=loops,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        z_min=z_min,
        z_max=z_max,
        spacing=grid_spacing,
    )
    d2 = grid.values
    z = _z_coordinates(grid)
    dz = np.abs(z) - height / 2.0
    sdf = _rounded_intersection(d2, dz, cap_radius)
    sdf_mesh = _mesh_from_sdf(sdf, grid.origin, grid.spacing)
    return sdf_mesh


def loft_sdf(
    profiles: Sequence[Profile2D],
    path: Path3D | PolyPath | Sequence[Sequence[float]] | None = None,
    positions: Sequence[float] | None = None,
    cap_radius: float = 0.0,
    grid_spacing: float = 0.2,
    padding: float | None = None,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> Mesh:
    """Loft profiles and apply SDF-based rounded endcaps."""

    if len(profiles) < 2:
        raise ValueError("loft_sdf requires at least two profiles.")
    cap_radius = float(max(cap_radius, 0.0))
    grid_spacing = float(grid_spacing)
    if grid_spacing <= 0:
        raise ValueError("grid_spacing must be positive.")

    if path is not None:
        positions = _resolve_positions(path, len(profiles))[:, 2]
    if positions is None:
        positions = np.linspace(-0.5, 0.5, len(profiles))
    positions = np.asarray(positions, dtype=float).reshape(-1)
    if positions.shape[0] != len(profiles):
        raise ValueError("positions must match the number of profiles.")

    loop_sets = [
        _profile_loops(profile, segments_per_circle, bezier_samples, enforce_winding=True)
        for profile in profiles
    ]
    bbox_min, bbox_max = _loops_bbox([loop for loops in loop_sets for loop in loops if loop.size > 0])
    base_pad = cap_radius if padding is None else float(padding)
    pad = base_pad + 3.0 * grid_spacing
    bbox_min = bbox_min - pad
    bbox_max = bbox_max + pad
    z_min = float(np.min(positions)) - pad
    z_max = float(np.max(positions)) + pad

    grid = _sdf_grid_from_loops(
        loops=loop_sets[0],
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        z_min=z_min,
        z_max=z_max,
        spacing=grid_spacing,
    )
    z = _z_coordinates(grid)
    body = np.full_like(grid.values, np.inf)

    slice_thickness = _slice_half_thickness(positions)
    for pos, loops in zip(positions, loop_sets, strict=False):
        d2 = _d2_from_loops(grid, loops)
        dz = np.abs(z - pos) - slice_thickness
        slab = np.maximum(d2, dz)
        body = np.minimum(body, slab)

    sdf = body
    if cap_radius > 0:
        d2_bot = _d2_from_loops(grid, loop_sets[0])
        d2_top = _d2_from_loops(grid, loop_sets[-1])
        dz_bot = (positions.min() - z)
        dz_top = (z - positions.max())
        cap_bot = _rounded_intersection(d2_bot, dz_bot, cap_radius)
        cap_top = _rounded_intersection(d2_top, dz_top, cap_radius)
        sdf = np.maximum.reduce([body, cap_bot, cap_top])

    sdf_mesh = _mesh_from_sdf(sdf, grid.origin, grid.spacing)
    return sdf_mesh


def _resolve_positions(
    path: Path3D | PolyPath | Sequence[Sequence[float]],
    count: int,
) -> np.ndarray:
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


def _rounded_intersection(d2: np.ndarray, dz: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return np.maximum(d2, dz)
    ax = np.maximum(d2, 0.0)
    az = np.maximum(dz, 0.0)
    outside = np.sqrt(ax * ax + az * az)
    inside = np.minimum(np.maximum(d2, dz), 0.0)
    return outside + inside - radius


def _sdf_grid_from_loops(
    loops: list[np.ndarray],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    z_min: float,
    z_max: float,
    spacing: float,
) -> SdfGrid:
    xs = _axis_samples(bbox_min[0], bbox_max[0], spacing)
    ys = _axis_samples(bbox_min[1], bbox_max[1], spacing)
    zs = _axis_samples(z_min, z_max, spacing)
    xv, yv, zv = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack([xv.ravel(), yv.ravel()])
    d2 = _d2_polygon(points, loops).reshape(xv.shape)
    origin = np.array([xs[0], ys[0], zs[0]], dtype=float)
    return SdfGrid(values=d2, origin=origin, spacing=spacing)


def _d2_from_loops(grid: SdfGrid, loops: list[np.ndarray]) -> np.ndarray:
    xv, yv, _ = _grid_coords(grid)
    points = np.column_stack([xv.ravel(), yv.ravel()])
    return _d2_polygon(points, loops).reshape(xv.shape)


def _grid_coords(grid: SdfGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = grid.values.shape
    xs = grid.origin[0] + np.arange(shape[0]) * grid.spacing
    ys = grid.origin[1] + np.arange(shape[1]) * grid.spacing
    zs = grid.origin[2] + np.arange(shape[2]) * grid.spacing
    return np.meshgrid(xs, ys, zs, indexing="ij")


def _z_coordinates(grid: SdfGrid) -> np.ndarray:
    _, _, zv = _grid_coords(grid)
    return zv


def _axis_samples(min_val: float, max_val: float, spacing: float) -> np.ndarray:
    count = int(np.ceil((max_val - min_val) / spacing)) + 1
    return np.linspace(min_val, max_val, count)


def _loops_bbox(loops: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    points = np.vstack([loop for loop in loops if loop.size > 0])
    return points.min(axis=0), points.max(axis=0)


def _slice_half_thickness(positions: np.ndarray) -> float:
    if positions.size < 2:
        return 0.5
    diffs = np.diff(np.sort(positions))
    return float(np.min(diffs) / 2.0) if diffs.size else 0.5


def _d2_polygon(points: np.ndarray, loops: list[np.ndarray]) -> np.ndarray:
    distances = _distance_to_segments(points, loops)
    inside = _even_odd_inside(points, loops)
    signed = distances.copy()
    signed[inside] *= -1.0
    return signed


def _distance_to_segments(points: np.ndarray, loops: list[np.ndarray]) -> np.ndarray:
    min_dist = np.full(points.shape[0], np.inf, dtype=float)
    for loop in loops:
        count = loop.shape[0]
        for i in range(count):
            a = loop[i]
            b = loop[(i + 1) % count]
            dist = _point_segment_distance(points, a, b)
            min_dist = np.minimum(min_dist, dist)
    return min_dist


def _point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ap = points - a
    denom = np.dot(ab, ab)
    if denom == 0:
        return np.linalg.norm(ap, axis=1)
    t = np.clip((ap @ ab) / denom, 0.0, 1.0)
    closest = a + np.outer(t, ab)
    return np.linalg.norm(points - closest, axis=1)


def _even_odd_inside(points: np.ndarray, loops: list[np.ndarray]) -> np.ndarray:
    inside = np.zeros(points.shape[0], dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    for loop in loops:
        n = loop.shape[0]
        x0 = loop[:, 0]
        y0 = loop[:, 1]
        x1 = np.roll(x0, -1)
        y1 = np.roll(y0, -1)
        cond = ((y0 <= y[:, None]) & (y1 > y[:, None])) | ((y1 <= y[:, None]) & (y0 > y[:, None]))
        xints = (x1 - x0) * (y[:, None] - y0) / (y1 - y0 + 1e-12) + x0
        crossings = cond & (x[:, None] < xints)
        inside ^= crossings.sum(axis=1) % 2 == 1
    return inside


def _mesh_from_sdf(values: np.ndarray, origin: np.ndarray, spacing: float) -> Mesh:
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("scikit-image is required for SDF meshing.") from exc

    verts, faces, normals, _ = marching_cubes(values, level=0.0, spacing=(spacing, spacing, spacing))
    verts = verts + origin
    return Mesh(verts, faces.astype(int))


__all__ = ["extrude_sdf", "loft_sdf"]
