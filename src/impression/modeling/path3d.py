from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np

from impression.mesh import Polyline
from impression.modeling._color import _normalize_color


def _require_vec3(value: Sequence[float], label: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).reshape(3)
    except Exception as exc:
        raise ValueError(f"{label} must be a 3D coordinate.") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite.")
    return arr


@dataclass(frozen=True)
class Line3D:
    start: np.ndarray
    end: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _require_vec3(self.start, "start"))
        object.__setattr__(self, "end", _require_vec3(self.end, "end"))

    def sample(self) -> np.ndarray:
        return np.vstack([self.start, self.end])


@dataclass(frozen=True)
class Arc3D:
    center: np.ndarray
    radius: float
    start_angle_deg: float
    end_angle_deg: float
    normal: np.ndarray
    clockwise: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _require_vec3(self.center, "center"))
        object.__setattr__(self, "normal", _require_vec3(self.normal, "normal"))
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be positive.")
        norm = np.linalg.norm(self.normal)
        if norm == 0:
            raise ValueError("normal must be non-zero.")
        object.__setattr__(self, "normal", self.normal / norm)

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
        u, v = _plane_basis(self.normal)
        x = self.center + (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v) * self.radius
        return x


@dataclass(frozen=True)
class Bezier3D:
    p0: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    p3: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "p0", _require_vec3(self.p0, "p0"))
        object.__setattr__(self, "p1", _require_vec3(self.p1, "p1"))
        object.__setattr__(self, "p2", _require_vec3(self.p2, "p2"))
        object.__setattr__(self, "p3", _require_vec3(self.p3, "p3"))

    def sample(self, samples: int) -> np.ndarray:
        samples = max(int(samples), 2)
        t = np.linspace(0.0, 1.0, samples, endpoint=True).reshape(-1, 1)
        a = (1 - t) ** 3
        b = 3 * (1 - t) ** 2 * t
        c = 3 * (1 - t) * t**2
        d = t**3
        return a * self.p0 + b * self.p1 + c * self.p2 + d * self.p3


Segment3D = Line3D | Arc3D | Bezier3D


@dataclass
class Path3D:
    segments: List[Segment3D] = field(default_factory=list)
    closed: bool = False
    color: tuple[float, float, float, float] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_points(cls, points: Iterable[Sequence[float]], closed: bool = False) -> "Path3D":
        pts = [_require_vec3(p, "point") for p in points]
        if len(pts) < 2:
            raise ValueError("Path3D requires at least two points.")
        segments: list[Segment3D] = [Line3D(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        if closed and not np.allclose(pts[0], pts[-1]):
            segments.append(Line3D(pts[-1], pts[0]))
        return cls(segments=segments, closed=closed)

    def sample(
        self,
        segments_per_circle: int = 64,
        bezier_samples: int = 32,
    ) -> np.ndarray:
        if not self.segments:
            return np.zeros((0, 3), dtype=float)
        points = []
        for idx, segment in enumerate(self.segments):
            if isinstance(segment, Line3D):
                seg_points = segment.sample()
            elif isinstance(segment, Arc3D):
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
        segments_per_circle: int = 64,
        bezier_samples: int = 32,
    ) -> Polyline:
        pts = self.sample(segments_per_circle=segments_per_circle, bezier_samples=bezier_samples)
        return Polyline(pts, closed=self.closed, color=self.color)

    def with_color(self, color: Sequence[float] | str | None) -> "Path3D":
        if color is None:
            self.color = None
            return self
        self.color = _normalize_color(color)
        return self


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, axis))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, normal)
    norm = np.linalg.norm(u)
    if norm == 0:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = u / norm
    v = np.cross(normal, u)
    return u, v


__all__ = ["Line3D", "Arc3D", "Bezier3D", "Path3D"]
