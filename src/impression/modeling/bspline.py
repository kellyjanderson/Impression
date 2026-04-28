from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

ClosurePolicy = Literal["open", "closed", "periodic"]


def _require_point(value: Sequence[float], *, dim: int, label: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).reshape(dim)
    except Exception as exc:
        raise ValueError(f"{label} must be a {dim}D coordinate.") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite.")
    return arr


def _normalize_control_points(
    values: Sequence[Sequence[float]],
    *,
    dim: int,
) -> tuple[np.ndarray, ...]:
    points = tuple(_require_point(value, dim=dim, label="control point") for value in values)
    if len(points) < 2:
        raise ValueError("B-spline curves require at least two control points.")
    return points


def _normalize_degree(value: int) -> int:
    degree = int(value)
    if degree < 1:
        raise ValueError("degree must be >= 1.")
    return degree


def _normalize_knots(values: Sequence[float]) -> tuple[float, ...]:
    knots = tuple(float(v) for v in values)
    if len(knots) < 4:
        raise ValueError("knot vector is too short.")
    if not all(np.isfinite(v) for v in knots):
        raise ValueError("knot vector values must be finite.")
    if any(b < a for a, b in zip(knots, knots[1:])):
        raise ValueError("knot vector must be nondecreasing.")
    return knots


def _normalize_closure(value: str) -> ClosurePolicy:
    allowed: set[str] = {"open", "closed", "periodic"}
    closure = str(value)
    if closure not in allowed:
        raise ValueError("closure must be one of: open, closed, periodic.")
    return closure  # type: ignore[return-value]


def _validate_shape(
    control_points: tuple[np.ndarray, ...],
    degree: int,
    knots: tuple[float, ...],
) -> None:
    if len(control_points) <= degree:
        raise ValueError("control point count must be greater than degree.")
    expected = len(control_points) + degree + 1
    if len(knots) != expected:
        raise ValueError(
            "knot vector length must equal control_point_count + degree + 1."
        )


@dataclass(frozen=True)
class BSpline2D:
    control_points: tuple[np.ndarray, ...]
    degree: int
    knots: tuple[float, ...]
    closure: ClosurePolicy = "open"

    def __init__(
        self,
        control_points: Sequence[Sequence[float]],
        degree: int,
        knots: Sequence[float],
        closure: ClosurePolicy = "open",
    ) -> None:
        points = _normalize_control_points(control_points, dim=2)
        deg = _normalize_degree(degree)
        knot_vector = _normalize_knots(knots)
        close = _normalize_closure(closure)
        _validate_shape(points, deg, knot_vector)
        object.__setattr__(self, "control_points", points)
        object.__setattr__(self, "degree", deg)
        object.__setattr__(self, "knots", knot_vector)
        object.__setattr__(self, "closure", close)


@dataclass(frozen=True)
class BSpline3D:
    control_points: tuple[np.ndarray, ...]
    degree: int
    knots: tuple[float, ...]
    closure: ClosurePolicy = "open"

    def __init__(
        self,
        control_points: Sequence[Sequence[float]],
        degree: int,
        knots: Sequence[float],
        closure: ClosurePolicy = "open",
    ) -> None:
        points = _normalize_control_points(control_points, dim=3)
        deg = _normalize_degree(degree)
        knot_vector = _normalize_knots(knots)
        close = _normalize_closure(closure)
        _validate_shape(points, deg, knot_vector)
        object.__setattr__(self, "control_points", points)
        object.__setattr__(self, "degree", deg)
        object.__setattr__(self, "knots", knot_vector)
        object.__setattr__(self, "closure", close)


__all__ = ["BSpline2D", "BSpline3D", "ClosurePolicy"]
