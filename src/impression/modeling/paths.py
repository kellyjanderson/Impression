from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from impression.mesh import Polyline


@dataclass
class Path:
    """Represents a polyline path that can be sampled or converted to splines."""

    points: List[np.ndarray]
    closed: bool = False

    @classmethod
    def from_points(cls, points: Iterable[Sequence[float]], closed: bool = False) -> "Path":
        pts = [np.asarray(p, dtype=float) for p in points]
        if len(pts) < 2:
            raise ValueError("Path requires at least two points.")
        return cls(points=pts, closed=closed)

    def length(self) -> float:
        pts = self._effective_points()
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    def sample(self, n_points: int) -> np.ndarray:
        """Return n_points samples along the path."""

        if n_points < 2:
            raise ValueError("sample requires at least two points.")
        pts = self._effective_points()
        distances = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        total = distances[-1]
        targets = np.linspace(0.0, total, n_points)
        sampled = []
        idx = 0
        for t in targets:
            while idx < len(distances) - 2 and distances[idx + 1] < t:
                idx += 1
            t0, t1 = distances[idx], distances[idx + 1]
            p0, p1 = pts[idx], pts[idx + 1]
            if t1 == t0:
                sampled.append(p0)
            else:
                alpha = (t - t0) / (t1 - t0)
                sampled.append((1 - alpha) * p0 + alpha * p1)
        return np.asarray(sampled)

    def to_polyline(self) -> Polyline:
        pts = self._effective_points()
        return Polyline(pts, closed=self.closed)

    def to_spline(self, n_samples: int = 200) -> Polyline:
        pts = self.sample(n_samples)
        return Polyline(pts, closed=self.closed)

    def _effective_points(self) -> np.ndarray:
        pts = np.asarray(self.points)
        if self.closed and not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        return pts
