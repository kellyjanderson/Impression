from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np

from impression.mesh import Polyline

from ._color import _normalize_color


def _to_vec2(value: Sequence[float]) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(2)
    return arr


def _require_vec2(value: Sequence[float], label: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).reshape(2)
    except Exception as exc:
        raise ValueError(f"{label} must be a 2D coordinate.") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite.")
    return arr


@dataclass(frozen=True)
class Line2D:
    start: np.ndarray
    end: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _require_vec2(self.start, "start"))
        object.__setattr__(self, "end", _require_vec2(self.end, "end"))

    def sample(self) -> np.ndarray:
        return np.vstack([self.start, self.end])


@dataclass(frozen=True)
class Arc2D:
    center: np.ndarray
    radius: float
    start_angle_deg: float
    end_angle_deg: float
    clockwise: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _require_vec2(self.center, "center"))
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be positive.")

    def sample(self, segments_per_circle: int) -> np.ndarray:
        if segments_per_circle < 3:
            raise ValueError("segments_per_circle must be >= 3.")
        start = np.deg2rad(self.start_angle_deg)
        end = np.deg2rad(self.end_angle_deg)
        if self.clockwise:
            if end > start:
                end -= 2 * np.pi
        else:
            if end < start:
                end += 2 * np.pi
        span = abs(end - start)
        steps = max(int(np.ceil(segments_per_circle * (span / (2 * np.pi)))), 2)
        angles = np.linspace(start, end, steps, endpoint=True)
        x = self.center[0] + self.radius * np.cos(angles)
        y = self.center[1] + self.radius * np.sin(angles)
        return np.column_stack([x, y])


@dataclass(frozen=True)
class Bezier2D:
    p0: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    p3: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "p0", _require_vec2(self.p0, "p0"))
        object.__setattr__(self, "p1", _require_vec2(self.p1, "p1"))
        object.__setattr__(self, "p2", _require_vec2(self.p2, "p2"))
        object.__setattr__(self, "p3", _require_vec2(self.p3, "p3"))

    def sample(self, samples: int) -> np.ndarray:
        samples = max(int(samples), 2)
        t = np.linspace(0.0, 1.0, samples, endpoint=True)
        t = t.reshape(-1, 1)
        a = (1 - t) ** 3
        b = 3 * (1 - t) ** 2 * t
        c = 3 * (1 - t) * t**2
        d = t**3
        return a * self.p0 + b * self.p1 + c * self.p2 + d * self.p3


Segment2D = Line2D | Arc2D | Bezier2D


@dataclass
class Path2D:
    segments: List[Segment2D] = field(default_factory=list)
    closed: bool = False
    color: tuple[float, float, float, float] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_points(cls, points: Iterable[Sequence[float]], closed: bool = True) -> "Path2D":
        pts = [_require_vec2(p, "point") for p in points]
        if len(pts) < 2:
            raise ValueError("Path2D requires at least two points.")
        segments = [Line2D(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        if closed and not np.allclose(pts[0], pts[-1]):
            segments.append(Line2D(pts[-1], pts[0]))
        return cls(segments=segments, closed=closed)

    def sample(
        self,
        segments_per_circle: int = 64,
        bezier_samples: int = 32,
    ) -> np.ndarray:
        if not self.segments:
            return np.zeros((0, 2), dtype=float)
        points = []
        for idx, segment in enumerate(self.segments):
            if isinstance(segment, Line2D):
                seg_points = segment.sample()
            elif isinstance(segment, Arc2D):
                seg_points = segment.sample(segments_per_circle)
            else:
                seg_points = segment.sample(bezier_samples)
            if idx > 0 and seg_points.shape[0] > 0:
                seg_points = seg_points[1:]
            points.append(seg_points)
        pts = np.vstack(points)
        if self.closed and pts.shape[0] > 0 and not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        return pts

    def to_polyline(
        self,
        z: float = 0.0,
        segments_per_circle: int = 64,
        bezier_samples: int = 32,
    ) -> Polyline:
        pts = self.sample(segments_per_circle=segments_per_circle, bezier_samples=bezier_samples)
        pts3 = np.column_stack([pts, np.full((pts.shape[0], 1), float(z))])
        return Polyline(pts3, closed=self.closed, color=self.color)

    def with_color(self, color: Sequence[float] | str | None) -> "Path2D":
        if color is None:
            self.color = None
            return self
        self.color = _normalize_color(color)
        return self


def _line_intersection(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> np.ndarray | None:
    """Return the intersection point for two 2D lines (point+direction)."""
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / cross
    return p1 + d1 * t


def round_corners(
    points: Iterable[Sequence[float]],
    radius: float,
    closed: bool = True,
    clamp: bool = True,
) -> Path2D:
    """Round sharp polyline corners with true arcs."""
    pts = [_require_vec2(p, "point") for p in points]
    if len(pts) < (3 if closed else 2):
        raise ValueError("round_corners requires at least three points for closed paths.")
    if closed and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < (3 if closed else 2):
        raise ValueError("round_corners requires at least three points for closed paths.")
    if radius <= 0:
        raise ValueError("radius must be positive.")

    n = len(pts)
    corners: list[dict[str, np.ndarray | float | bool] | None] = [None] * n
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            continue
        prev = pts[i - 1 if i > 0 else n - 1]
        curr = pts[i]
        nxt = pts[(i + 1) % n]
        v1 = curr - prev
        v2 = nxt - curr
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 1e-9 or len2 < 1e-9:
            continue
        u1 = v1 / len1
        u2 = v2 / len2
        dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
        angle = float(np.arccos(dot))
        if angle < 1e-6 or abs(np.pi - angle) < 1e-6:
            continue

        max_r = min(len1, len2) / max(np.tan(angle / 2.0), 1e-9)
        r = float(radius)
        if r > max_r:
            if clamp:
                r = max_r
            else:
                continue
        if r <= 0:
            continue

        cross = u1[0] * u2[1] - u1[1] * u2[0]
        if abs(cross) < 1e-9:
            continue
        sign = 1.0 if cross > 0 else -1.0
        n1 = sign * np.array([-u1[1], u1[0]])
        n2 = sign * np.array([-u2[1], u2[0]])
        p1 = curr + n1 * r
        p2 = curr + n2 * r
        center = _line_intersection(p1, u1, p2, u2)
        if center is None:
            continue

        t1 = curr + u1 * np.dot(u1, center - curr)
        t2 = curr + u2 * np.dot(u2, center - curr)
        start = float(np.degrees(np.arctan2(t1[1] - center[1], t1[0] - center[0])))
        end = float(np.degrees(np.arctan2(t2[1] - center[1], t2[0] - center[0])))
        corners[i] = {
            "t1": t1,
            "t2": t2,
            "center": center,
            "radius": float(np.linalg.norm(t1 - center)),
            "start": start,
            "end": end,
            "clockwise": cross < 0,
        }

    segments: list[Segment2D] = []
    if closed:
        start_point = corners[0]["t1"] if corners[0] is not None else pts[0]
        prev_anchor = start_point
        for i in range(n):
            corner = corners[i]
            if corner is None:
                line_end = pts[i]
                if not np.allclose(prev_anchor, line_end):
                    segments.append(Line2D(prev_anchor, line_end))
                prev_anchor = line_end
                continue
            t1 = corner["t1"]
            if not np.allclose(prev_anchor, t1):
                segments.append(Line2D(prev_anchor, t1))
            segments.append(
                Arc2D(
                    center=corner["center"],
                    radius=corner["radius"],
                    start_angle_deg=corner["start"],
                    end_angle_deg=corner["end"],
                    clockwise=bool(corner["clockwise"]),
                )
            )
            prev_anchor = corner["t2"]
        if not np.allclose(prev_anchor, start_point):
            segments.append(Line2D(prev_anchor, start_point))
        return Path2D(segments=segments, closed=True)

    prev_anchor = pts[0]
    for i in range(1, n - 1):
        corner = corners[i]
        if corner is None:
            line_end = pts[i]
            if not np.allclose(prev_anchor, line_end):
                segments.append(Line2D(prev_anchor, line_end))
            prev_anchor = line_end
            continue
        t1 = corner["t1"]
        if not np.allclose(prev_anchor, t1):
            segments.append(Line2D(prev_anchor, t1))
        segments.append(
            Arc2D(
                center=corner["center"],
                radius=corner["radius"],
                start_angle_deg=corner["start"],
                end_angle_deg=corner["end"],
                clockwise=bool(corner["clockwise"]),
            )
        )
        prev_anchor = corner["t2"]

    if not np.allclose(prev_anchor, pts[-1]):
        segments.append(Line2D(prev_anchor, pts[-1]))
    return Path2D(segments=segments, closed=False)


def round_path(path: Path2D, radius: float, clamp: bool = True) -> Path2D:
    """Round corners of a Path2D using true arcs."""
    if all(isinstance(seg, Line2D) for seg in path.segments):
        pts = []
        for segment in path.segments:
            pts.append(segment.start)
        pts.append(path.segments[-1].end)
        return round_corners(pts, radius=radius, closed=path.closed, clamp=clamp)
    sampled = path.sample()
    return round_corners(sampled, radius=radius, closed=path.closed, clamp=clamp)


@dataclass
class Profile2D:
    outer: Path2D
    holes: List[Path2D] = field(default_factory=list)
    color: tuple[float, float, float, float] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.outer.closed:
            raise ValueError("Profile2D outer path must be closed.")
        for hole in self.holes:
            if not hole.closed:
                raise ValueError("Profile2D hole paths must be closed.")

    def to_polylines(
        self,
        z: float = 0.0,
        segments_per_circle: int = 64,
        bezier_samples: int = 32,
    ) -> list[Polyline]:
        polylines = [self.outer.to_polyline(z=z, segments_per_circle=segments_per_circle, bezier_samples=bezier_samples)]
        if self.color is not None:
            polylines[0].color = self.color
        for hole in self.holes:
            poly = hole.to_polyline(z=z, segments_per_circle=segments_per_circle, bezier_samples=bezier_samples)
            if self.color is not None:
                poly.color = self.color
            polylines.append(poly)
        return polylines

    def with_color(self, color: Sequence[float] | str | None) -> "Profile2D":
        if color is None:
            self.color = None
            return self
        self.color = _normalize_color(color)
        return self


def make_rect(
    size: Sequence[float] = (1.0, 1.0),
    center: Sequence[float] = (0.0, 0.0),
    color: Sequence[float] | str | None = None,
) -> Profile2D:
    sx, sy = float(size[0]), float(size[1])
    if sx <= 0 or sy <= 0:
        raise ValueError("size must be positive.")
    cx, cy = _to_vec2(center)
    hx, hy = sx / 2.0, sy / 2.0
    points = [
        (cx - hx, cy - hy),
        (cx + hx, cy - hy),
        (cx + hx, cy + hy),
        (cx - hx, cy + hy),
    ]
    outer = Path2D.from_points(points, closed=True)
    profile = Profile2D(outer=outer)
    if color is not None:
        profile.with_color(color)
    return profile


def make_circle(
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0),
    color: Sequence[float] | str | None = None,
) -> Profile2D:
    if radius <= 0:
        raise ValueError("radius must be positive.")
    center_vec = _to_vec2(center)
    arc = Arc2D(center=center_vec, radius=float(radius), start_angle_deg=0.0, end_angle_deg=360.0)
    outer = Path2D(segments=[arc], closed=True)
    profile = Profile2D(outer=outer)
    if color is not None:
        profile.with_color(color)
    return profile


def make_ngon(
    sides: int = 6,
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0),
    color: Sequence[float] | str | None = None,
    *,
    side_length: float | None = None,
) -> Profile2D:
    sides = int(sides)
    if sides < 3:
        raise ValueError("sides must be >= 3.")
    if side_length is not None:
        inferred = float(side_length) / (2.0 * np.sin(np.pi / sides))
        if radius != 0.5 and not np.isclose(radius, inferred):
            raise ValueError("Specify either radius or side_length, not both.")
        radius = inferred
    radius = float(radius)
    if radius <= 0:
        raise ValueError("radius must be positive.")
    center_vec = _to_vec2(center)
    angles = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)]) * radius
    points = points + center_vec
    outer = Path2D.from_points(points, closed=True)
    profile = Profile2D(outer=outer)
    if color is not None:
        profile.with_color(color)
    return profile


def make_polygon(
    points: Iterable[Sequence[float]],
    color: Sequence[float] | str | None = None,
) -> Profile2D:
    pts = list(points)
    if len(pts) < 3:
        raise ValueError("make_polygon requires at least three points.")
    outer = Path2D.from_points(pts, closed=True)
    profile = Profile2D(outer=outer)
    if color is not None:
        profile.with_color(color)
    return profile


def make_polyline(
    points: Iterable[Sequence[float]],
    closed: bool = False,
    color: Sequence[float] | str | None = None,
) -> Path2D:
    path = Path2D.from_points(points, closed=closed)
    if color is not None:
        path.with_color(color)
    return path


__all__ = [
    "Arc2D",
    "Bezier2D",
    "Line2D",
    "Path2D",
    "Profile2D",
    "make_circle",
    "make_ngon",
    "make_polyline",
    "make_polygon",
    "make_rect",
]
