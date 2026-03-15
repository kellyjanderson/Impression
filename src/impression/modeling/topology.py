from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

from impression.modeling.drawing2d import Path2D
from ._color import _normalize_color


def _normalize_loop_points(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 0:
        return pts
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


@dataclass(frozen=True)
class Loop:
    """A closed 2D loop in local plane coordinates."""

    points: np.ndarray

    def __post_init__(self) -> None:
        pts = _normalize_loop_points(self.points)
        object.__setattr__(self, "points", pts)

    @property
    def area(self) -> float:
        return signed_area(self.points)

    @property
    def is_clockwise(self) -> bool:
        return self.area < 0.0

    @property
    def perimeter(self) -> float:
        if self.points.shape[0] < 2:
            return 0.0
        closed = np.vstack([self.points, self.points[0]])
        return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        if self.points.shape[0] == 0:
            return (0.0, 0.0, 0.0, 0.0)
        mins = self.points.min(axis=0)
        maxs = self.points.max(axis=0)
        return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]))

    @property
    def centroid(self) -> np.ndarray:
        if self.points.shape[0] == 0:
            return np.zeros(2, dtype=float)
        return self.points.mean(axis=0)

    def with_winding(self, clockwise: bool) -> "Loop":
        return Loop(ensure_winding(self.points, clockwise=clockwise))

    def resampled(self, count: int) -> "Loop":
        return Loop(resample_loop(self.points, count))

    def anchored(self) -> "Loop":
        return Loop(anchor_loop(self.points))

    def to_path(self) -> Path2D:
        return Path2D.from_points(self.points, closed=True)


@dataclass(frozen=True)
class Region:
    """A planar solid region with one outer loop and zero or more holes."""

    outer: Loop
    holes: tuple[Loop, ...] = ()

    def normalized(self) -> "Region":
        outer = self.outer.with_winding(clockwise=False)
        holes = tuple(hole.with_winding(clockwise=True) for hole in self.holes)
        return Region(outer=outer, holes=holes)

    def is_valid(self) -> bool:
        normalized = self.normalized()
        for hole in normalized.holes:
            if hole.points.shape[0] < 3:
                return False
            if not point_in_polygon(hole.points[0], normalized.outer.points):
                return False
        return normalized.outer.points.shape[0] >= 3

@dataclass(frozen=True)
class Section:
    """A planar collection of one or more disconnected regions."""

    regions: tuple[Region, ...] = field(default_factory=tuple)
    color: tuple[float, float, float, float] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def normalized(self) -> "Section":
        return Section(
            tuple(region.normalized() for region in self.regions),
            color=self.color,
            metadata=dict(self.metadata),
        )

    def with_color(self, color: Sequence[float] | str | None) -> "Section":
        rgba = None if color is None else _normalize_color(color)
        return Section(
            regions=self.regions,
            color=rgba,
            metadata=dict(self.metadata),
        )


def as_section(
    shape: Section | Region | Path2D | object,
    *,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> Section:
    """Normalize a planar input shape into a topology-native Section."""

    if isinstance(shape, Section):
        return shape.normalized()
    if isinstance(shape, Region):
        return Section((shape.normalized(),))
    if isinstance(shape, Path2D):
        if not shape.closed:
            raise ValueError("Path2D must be closed for planar solid operations.")
        pts = _normalize_loop_points(shape.sample(segments_per_circle=segments_per_circle, bezier_samples=bezier_samples))
        if pts.shape[0] < 3:
            return Section(())
        return Section(
            (Region(outer=Loop(ensure_winding(pts, clockwise=False))),),
            color=shape.color,
            metadata=dict(shape.metadata),
        ).normalized()
    if hasattr(shape, "outer") and hasattr(shape, "holes"):
        outer = getattr(shape, "outer")
        holes = getattr(shape, "holes")
        if not isinstance(outer, Path2D):
            raise TypeError("Expected .outer to be a closed Path2D.")
        if not outer.closed:
            raise ValueError("Outer path must be closed for planar solid operations.")
        loops = [loop_points(outer, segments_per_circle, bezier_samples)]
        for hole in holes:
            if not isinstance(hole, Path2D):
                raise TypeError("Expected hole paths to be Path2D values.")
            if not hole.closed:
                raise ValueError("Hole paths must be closed for planar solid operations.")
            loops.append(loop_points(hole, segments_per_circle, bezier_samples))
        if not loops or loops[0].shape[0] < 3:
            return Section(())
        region = Region(
            outer=Loop(ensure_winding(loops[0], clockwise=False)),
            holes=tuple(Loop(ensure_winding(loop, clockwise=True)) for loop in loops[1:]),
        ).normalized()
        color = getattr(shape, "color", None)
        metadata = dict(getattr(shape, "metadata", {}) or {})
        return Section((region,), color=color, metadata=metadata)
    raise TypeError("Expected Section, Region, closed Path2D, or shape with .outer/.holes Path2D loops.")


def as_sections(
    shapes: Iterable[Section | Region | Path2D | object],
    *,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> list[Section]:
    """Normalize multiple planar input shapes into topology-native Sections."""

    return [
        as_section(
            shape,
            segments_per_circle=segments_per_circle,
            bezier_samples=bezier_samples,
        )
        for shape in shapes
    ]


def signed_area(points: np.ndarray) -> float:
    pts = _normalize_loop_points(points)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ensure_winding(points: np.ndarray, clockwise: bool) -> np.ndarray:
    pts = _normalize_loop_points(points)
    if pts.shape[0] < 3:
        return pts
    area = signed_area(pts)
    is_cw = area < 0.0
    if is_cw != clockwise:
        return pts[::-1].copy()
    return pts


def loop_points(
    path: Path2D,
    segments_per_circle: int,
    bezier_samples: int,
) -> np.ndarray:
    pts = path.sample(segments_per_circle=segments_per_circle, bezier_samples=bezier_samples)
    return _normalize_loop_points(pts)


def profile_loops(
    shape: Section | Region | Path2D | object,
    segments_per_circle: int,
    bezier_samples: int,
    enforce_winding: bool = True,
) -> list[np.ndarray]:
    section = as_section(
        shape,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    ).normalized()
    if len(section.regions) != 1:
        raise ValueError("profile_loops requires one connected region.")
    region = section.regions[0]
    loops: list[np.ndarray] = []
    outer = _normalize_loop_points(region.outer.points)
    if enforce_winding:
        outer = ensure_winding(outer, clockwise=False)
    loops.append(outer)

    for hole in region.holes:
        pts = _normalize_loop_points(hole.points)
        if enforce_winding:
            pts = ensure_winding(pts, clockwise=True)
        loops.append(pts)
    return loops


def triangulate_profile(
    shape: Section | Region | Path2D | object,
    segments_per_circle: int,
    bezier_samples: int,
    enforce_winding: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    loops = profile_loops(
        shape,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
        enforce_winding=enforce_winding,
    )
    vertices, faces = triangulate_loops(loops)
    return vertices, faces, loops


def triangulate_loops(loops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not loops:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=int)
    try:
        import mapbox_earcut as earcut
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("mapbox_earcut is required for profile triangulation.") from exc

    normalized_loops = [_normalize_loop_points(loop) for loop in loops]
    vertices = np.vstack(normalized_loops).astype(np.float32)
    ring_ends = []
    offset = 0
    for pts in normalized_loops:
        offset += pts.shape[0]
        ring_ends.append(offset)
    ring_end_indices = np.asarray(ring_ends, dtype=np.uint32)
    indices = earcut.triangulate_float32(vertices, ring_end_indices)
    faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    if not _triangulation_covers_loop_boundaries(faces, normalized_loops):
        jittered_vertices = _jitter_vertices_for_triangulation(normalized_loops)
        indices = earcut.triangulate_float32(jittered_vertices, ring_end_indices)
        faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    return vertices.astype(float), faces


def _triangulation_covers_loop_boundaries(
    faces: np.ndarray,
    loops: list[np.ndarray],
) -> bool:
    edge_set: set[tuple[int, int]] = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        edge_set.add((a, b) if a < b else (b, a))
        edge_set.add((b, c) if b < c else (c, b))
        edge_set.add((c, a) if c < a else (a, c))
    cursor = 0
    for loop in loops:
        count = int(loop.shape[0])
        if count < 3:
            return False
        for i in range(count):
            a = cursor + i
            b = cursor + ((i + 1) % count)
            key = (a, b) if a < b else (b, a)
            if key not in edge_set:
                return False
        cursor += count
    return True


def _jitter_vertices_for_triangulation(loops: list[np.ndarray]) -> np.ndarray:
    jittered: list[np.ndarray] = []
    for loop_index, loop in enumerate(loops):
        pts = np.asarray(loop, dtype=float)
        count = int(pts.shape[0])
        if count == 0:
            continue
        span = np.ptp(pts, axis=0)
        scale = max(float(np.hypot(span[0], span[1])), 1.0)
        eps = max(scale * 1e-3, 1e-6)
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False) + loop_index * 0.37
        perturb = np.column_stack((np.cos(angles), np.sin(angles))) * eps
        jittered.append((pts + perturb).astype(np.float32))
    if not jittered:
        return np.zeros((0, 2), dtype=np.float32)
    return np.vstack(jittered)


def resample_loop(points: np.ndarray, count: int) -> np.ndarray:
    if count < 3:
        raise ValueError("resample count must be >= 3.")
    pts = _normalize_loop_points(points)
    if pts.shape[0] < 2:
        raise ValueError("Loop requires at least two points.")
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


def loops_resampled(
    shape: Section | Region | Path2D | object,
    count: int,
    segments_per_circle: int,
    bezier_samples: int,
    enforce_winding: bool = True,
) -> list[np.ndarray]:
    loops = profile_loops(
        shape,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
        enforce_winding=enforce_winding,
    )
    return [resample_loop(loop, count) for loop in loops]


def anchor_loop(loop: np.ndarray) -> np.ndarray:
    pts = _normalize_loop_points(loop)
    if pts.shape[0] == 0:
        return pts

    centroid = pts.mean(axis=0)
    rel = pts - centroid
    angles = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2.0 * np.pi)
    min_angle = np.min(angles)
    candidates = np.where(np.isclose(angles, min_angle))[0]
    if candidates.size == 1:
        idx = int(candidates[0])
    else:
        radii = np.linalg.norm(rel[candidates], axis=1)
        max_radius = np.max(radii)
        radius_candidates = candidates[np.where(np.isclose(radii, max_radius))[0]]
        idx = int(np.min(radius_candidates))
    return np.roll(pts, -idx, axis=0)


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x = float(point[0])
    y = float(point[1])
    poly = _normalize_loop_points(polygon)
    if poly.shape[0] < 3:
        return False

    inside = False
    n = poly.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        if intersect:
            inside = not inside
        j = i
    return inside


def largest_loop(contours: list[np.ndarray]) -> np.ndarray:
    if not contours:
        raise ValueError("largest_loop requires at least one contour.")
    return max(contours, key=lambda c: abs(signed_area(c)))


def classify_loops(loops: list[np.ndarray], expected_holes: int | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    if not loops:
        raise ValueError("classify_loops requires at least one loop.")
    outer = largest_loop(loops)
    holes = [loop for loop in loops if loop is not outer]
    for hole in holes:
        if not point_in_polygon(_normalize_loop_points(hole)[0], outer):
            raise ValueError("Loop set contains disconnected geometry.")
    outer = ensure_winding(outer, clockwise=False)
    holes = [ensure_winding(hole, clockwise=True) for hole in holes]
    if expected_holes is not None and len(holes) != expected_holes:
        raise ValueError("Loop classification changed hole count.")
    return outer, holes


def inset_profile_loops(
    shape: Section | Region | Path2D | object,
    inset: float,
    *,
    join_type: str = "ROUND",
    hole_count: int = 0,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> list[np.ndarray]:
    """Inset a profile's outer/hole loops with topology validation.

    This helper centralizes profile inset behavior used by loft endcap
    generation so loop classification and containment checks stay in topology.
    """

    try:
        import pyclipper
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("pyclipper is required for profile inset operations.") from exc

    loops = profile_loops(
        shape,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
        enforce_winding=True,
    )
    if not loops:
        raise ValueError("Shape has no loops to inset.")
    if inset <= 0:
        return loops

    scale = 1_000_000.0
    join_key = "ROUND" if join_type.upper() in {"ROUND", "COVE"} else join_type.upper()
    jt = pyclipper.JT_ROUND if join_key == "ROUND" else pyclipper.JT_MITER

    def offset_single(path_pts: np.ndarray, delta: float) -> list[np.ndarray]:
        pco = pyclipper.PyclipperOffset(miter_limit=2.0, arc_tolerance=0.25 * scale)
        path = np.round(path_pts * scale).astype(np.int64).tolist()
        pco.AddPath(path, jt, pyclipper.ET_CLOSEDPOLYGON)
        result = pco.Execute(delta * scale)
        return [np.asarray(p, dtype=float) / scale for p in result]

    outer = loops[0]
    holes = loops[1:]

    outer_result = offset_single(outer, -inset)
    if len(outer_result) != 1:
        raise ValueError("endcap_amount too large for profile; inset collapsed.")
    outer = ensure_winding(outer_result[0], clockwise=False)

    inset_holes: list[np.ndarray] = []
    for hole in holes:
        hole_result = offset_single(hole, inset)
        if len(hole_result) != 1:
            raise ValueError("endcap_amount too large for profile; inset collapsed.")
        hole_loop = ensure_winding(hole_result[0], clockwise=True)
        if not point_in_polygon(hole_loop[0], outer):
            raise ValueError("endcap_amount too large for profile; hole collapsed.")
        inset_holes.append(hole_loop)

    if hole_count and len(inset_holes) != hole_count:
        raise ValueError("endcap_amount too large for profile; hole topology changed.")
    return [outer] + inset_holes


def sections_from_paths(paths: Iterable[Path2D]) -> list[Section]:
    """Assemble topology-native sections from a set of closed paths."""
    regions = regions_from_paths(paths)
    return [Section((region,)) for region in regions]


def regions_from_paths(paths: Iterable[Path2D]) -> list[Region]:
    """Assemble topology-native regions from a set of closed paths."""

    info = []
    for path in paths:
        pts = _normalize_loop_points(path.sample())
        if pts.shape[0] < 3:
            continue
        area = abs(signed_area(pts))
        info.append({"path": path, "pts": pts, "area": area, "holes": []})
    info.sort(key=lambda item: item["area"], reverse=True)

    outers = []
    for item in info:
        candidate = None
        for outer in outers:
            if point_in_polygon(item["pts"][0], outer["pts"]):
                if candidate is None or outer["area"] < candidate["area"]:
                    candidate = outer
        if candidate is None:
            outers.append(item)
        else:
            candidate["holes"].append(item["path"])
    regions: list[Region] = []
    for outer in outers:
        outer_path = outer["path"]
        holes_paths = outer.get("holes", [])
        outer_pts = _normalize_loop_points(outer_path.sample())
        outer_loop = Loop(ensure_winding(outer_pts, clockwise=False))
        hole_loops = []
        for hole_path in holes_paths:
            hole_pts = _normalize_loop_points(hole_path.sample())
            hole_loops.append(Loop(ensure_winding(hole_pts, clockwise=True)))
        regions.append(Region(outer=outer_loop, holes=tuple(hole_loops)).normalized())
    return regions


def minimum_cost_loop_assignment(
    source_loops: list[np.ndarray],
    target_loops: list[np.ndarray],
    *,
    area_weight: float = 0.1,
) -> tuple[int, ...]:
    """Return deterministic one-to-one loop correspondence.

    Primary score: centroid distance + weighted area delta.
    Tie-break order is lexicographic:
    1) lower total primary score
    2) lower total centroid distance
    3) lower total area delta
    4) lower target-index tuple in source order
    """

    if len(source_loops) != len(target_loops):
        raise ValueError("minimum_cost_loop_assignment requires equal source/target counts.")
    count = len(source_loops)
    if count == 0:
        return ()
    if count > 12:
        raise ValueError("Too many loops for deterministic assignment.")

    src_centroids = [np.asarray(loop, dtype=float).mean(axis=0) for loop in source_loops]
    dst_centroids = [np.asarray(loop, dtype=float).mean(axis=0) for loop in target_loops]
    src_areas = [abs(signed_area(loop)) for loop in source_loops]
    dst_areas = [abs(signed_area(loop)) for loop in target_loops]

    dist = np.zeros((count, count), dtype=float)
    area = np.zeros((count, count), dtype=float)
    primary = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(count):
            d = float(np.linalg.norm(src_centroids[i] - dst_centroids[j]))
            a = abs(src_areas[i] - dst_areas[j])
            dist[i, j] = d
            area[i, j] = a
            primary[i, j] = d + area_weight * a

    @lru_cache(maxsize=None)
    def solve(i: int, used_mask: int) -> tuple[tuple[float, float, float, tuple[int, ...]], tuple[int, ...]]:
        if i == count:
            zero = (0.0, 0.0, 0.0, ())
            return zero, ()

        best_score: tuple[float, float, float, tuple[int, ...]] | None = None
        best_order: tuple[int, ...] | None = None
        for j in range(count):
            if used_mask & (1 << j):
                continue
            child_score, child_order = solve(i + 1, used_mask | (1 << j))
            score = (
                primary[i, j] + child_score[0],
                dist[i, j] + child_score[1],
                area[i, j] + child_score[2],
                (j, *child_score[3]),
            )
            order = (j, *child_order)
            if best_score is None or score < best_score:
                best_score = score
                best_order = order

        if best_score is None or best_order is None:
            raise ValueError("Failed to compute deterministic loop assignment.")
        return best_score, best_order

    _, assignment = solve(0, 0)
    return assignment


def minimum_cost_subset_assignment(
    source_loops: list[np.ndarray],
    target_loops: list[np.ndarray],
    *,
    area_weight: float = 0.1,
) -> tuple[int, ...]:
    """Return deterministic source->target assignment where len(source) <= len(target)."""

    if len(source_loops) > len(target_loops):
        raise ValueError("subset assignment expects source count <= target count.")
    count = len(source_loops)
    if count == 0:
        return ()
    if len(target_loops) > 12:
        raise ValueError("Too many loops for deterministic subset assignment.")

    src_centroids = [np.asarray(loop, dtype=float).mean(axis=0) for loop in source_loops]
    dst_centroids = [np.asarray(loop, dtype=float).mean(axis=0) for loop in target_loops]
    src_areas = [abs(signed_area(loop)) for loop in source_loops]
    dst_areas = [abs(signed_area(loop)) for loop in target_loops]

    dist = np.zeros((count, len(target_loops)), dtype=float)
    area = np.zeros((count, len(target_loops)), dtype=float)
    primary = np.zeros((count, len(target_loops)), dtype=float)
    for i in range(count):
        for j in range(len(target_loops)):
            d = float(np.linalg.norm(src_centroids[i] - dst_centroids[j]))
            a = abs(src_areas[i] - dst_areas[j])
            dist[i, j] = d
            area[i, j] = a
            primary[i, j] = d + area_weight * a

    @lru_cache(maxsize=None)
    def solve(i: int, used_mask: int) -> tuple[tuple[float, float, float, tuple[int, ...]], tuple[int, ...]]:
        if i == count:
            zero = (0.0, 0.0, 0.0, ())
            return zero, ()

        best_score: tuple[float, float, float, tuple[int, ...]] | None = None
        best_order: tuple[int, ...] | None = None
        for j in range(len(target_loops)):
            if used_mask & (1 << j):
                continue
            child_score, child_order = solve(i + 1, used_mask | (1 << j))
            score = (
                primary[i, j] + child_score[0],
                dist[i, j] + child_score[1],
                area[i, j] + child_score[2],
                (j, *child_score[3]),
            )
            order = (j, *child_order)
            if best_score is None or score < best_score:
                best_score = score
                best_order = order
        if best_score is None or best_order is None:
            raise ValueError("Failed to compute deterministic subset assignment.")
        return best_score, best_order

    _, assignment = solve(0, 0)
    return assignment


def loop_bbox(loop: np.ndarray) -> tuple[float, float, float, float]:
    pts = np.asarray(loop, dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])


def bbox_area(box: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = min(ax1, bx1) - max(ax0, bx0)
    dy = min(ay1, by1) - max(ay0, by0)
    if dx <= 0 or dy <= 0:
        return 0.0
    return float(dx * dy)


def loop_span(loop: np.ndarray) -> float:
    x0, y0, x1, y1 = loop_bbox(loop)
    return float(np.hypot(x1 - x0, y1 - y0))


def stable_loop_transition(source_loop: np.ndarray, target_loop: np.ndarray) -> bool:
    source = np.asarray(source_loop, dtype=float)
    target = np.asarray(target_loop, dtype=float)
    src_centroid = source.mean(axis=0)
    dst_centroid = target.mean(axis=0)

    if point_in_polygon(src_centroid, target) or point_in_polygon(dst_centroid, source):
        return True
    if bbox_overlap_area(loop_bbox(source), loop_bbox(target)) > 1e-9:
        return True

    src_span = loop_span(source)
    dst_span = loop_span(target)
    dist = float(np.linalg.norm(src_centroid - dst_centroid))
    scale = max(src_span, dst_span, 1e-9)
    return dist <= (1.5 * scale)


def split_merge_ambiguous(a_loop: np.ndarray, b_loop: np.ndarray) -> bool:
    a = np.asarray(a_loop, dtype=float)
    b = np.asarray(b_loop, dtype=float)
    a_centroid = a.mean(axis=0)
    b_centroid = b.mean(axis=0)
    if point_in_polygon(a_centroid, b) or point_in_polygon(b_centroid, a):
        return True

    a_box = loop_bbox(a)
    b_box = loop_bbox(b)
    overlap = bbox_overlap_area(a_box, b_box)
    if overlap <= 0:
        return False
    min_box_area = max(min(bbox_area(a_box), bbox_area(b_box)), 1e-12)
    overlap_ratio = overlap / min_box_area
    return overlap_ratio >= 0.25


__all__ = [
    "Loop",
    "Region",
    "Section",
    "as_section",
    "as_sections",
    "anchor_loop",
    "classify_loops",
    "inset_profile_loops",
    "ensure_winding",
    "largest_loop",
    "loop_points",
    "loops_resampled",
    "point_in_polygon",
    "minimum_cost_loop_assignment",
    "minimum_cost_subset_assignment",
    "stable_loop_transition",
    "split_merge_ambiguous",
    "loop_bbox",
    "loop_span",
    "bbox_area",
    "bbox_overlap_area",
    "profile_loops",
    "regions_from_paths",
    "sections_from_paths",
    "resample_loop",
    "signed_area",
    "triangulate_loops",
    "triangulate_profile",
]
