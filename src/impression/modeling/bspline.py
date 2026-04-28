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


def _parameter_range(degree: int, knots: tuple[float, ...]) -> tuple[float, float]:
    start = float(knots[degree])
    end = float(knots[-degree - 1])
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError("parameter range must be finite and increasing.")
    return start, end


def _normalize_parameter(
    value: float,
    *,
    degree: int,
    knots: tuple[float, ...],
    closure: ClosurePolicy,
) -> float:
    start, end = _parameter_range(degree, knots)
    t = float(value)
    if not np.isfinite(t):
        raise ValueError("parameter value must be finite.")
    if closure == "periodic":
        span = end - start
        if span <= 0:
            raise ValueError("periodic curves require a positive parameter span.")
        return start + ((t - start) % span)
    if closure == "closed" and np.isclose(t, end):
        return start
    if t <= start:
        return start
    if t >= end:
        return end
    return t


def _find_span(
    *,
    degree: int,
    knots: tuple[float, ...],
    control_point_count: int,
    parameter: float,
) -> int:
    n = control_point_count - 1
    if parameter >= knots[n + 1]:
        return n
    if parameter <= knots[degree]:
        return degree
    low = degree
    high = n + 1
    mid = (low + high) // 2
    while parameter < knots[mid] or parameter >= knots[mid + 1]:
        if parameter < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def _basis_functions(span: int, parameter: float, degree: int, knots: tuple[float, ...]) -> np.ndarray:
    left = np.zeros(degree + 1, dtype=float)
    right = np.zeros(degree + 1, dtype=float)
    basis = np.zeros(degree + 1, dtype=float)
    basis[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = parameter - knots[span + 1 - j]
        right[j] = knots[span + j] - parameter
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            term = 0.0 if denom == 0.0 else basis[r] / denom
            basis[r] = saved + right[r + 1] * term
            saved = left[j - r] * term
        basis[j] = saved
    return basis


def _evaluate_point(
    *,
    control_points: tuple[np.ndarray, ...],
    degree: int,
    knots: tuple[float, ...],
    parameter: float,
) -> np.ndarray:
    span = _find_span(
        degree=degree,
        knots=knots,
        control_point_count=len(control_points),
        parameter=parameter,
    )
    basis = _basis_functions(span, parameter, degree, knots)
    point = np.zeros_like(control_points[0], dtype=float)
    for j in range(degree + 1):
        point = point + basis[j] * control_points[span - degree + j]
    return point


def _derivative_curve(
    *,
    control_points: tuple[np.ndarray, ...],
    degree: int,
    knots: tuple[float, ...],
) -> tuple[tuple[np.ndarray, ...], int, tuple[float, ...]]:
    derived: list[np.ndarray] = []
    for i in range(len(control_points) - 1):
        denom = knots[i + degree + 1] - knots[i + 1]
        if denom == 0.0:
            derived.append(np.zeros_like(control_points[i], dtype=float))
        else:
            derived.append((degree / denom) * (control_points[i + 1] - control_points[i]))
    return tuple(derived), degree - 1, knots[1:-1]


def _evaluate_derivative(
    *,
    control_points: tuple[np.ndarray, ...],
    degree: int,
    knots: tuple[float, ...],
    parameter: float,
) -> np.ndarray:
    if degree == 1:
        derived_points, derived_degree, derived_knots = _derivative_curve(
            control_points=control_points,
            degree=degree,
            knots=knots,
        )
        return _evaluate_point(
            control_points=derived_points,
            degree=derived_degree,
            knots=derived_knots,
            parameter=parameter,
        )
    derived_points, derived_degree, derived_knots = _derivative_curve(
        control_points=control_points,
        degree=degree,
        knots=knots,
    )
    return _evaluate_point(
        control_points=derived_points,
        degree=derived_degree,
        knots=derived_knots,
        parameter=parameter,
    )


def _sample_parameters(
    *,
    degree: int,
    knots: tuple[float, ...],
    closure: ClosurePolicy,
    n_samples: int,
) -> np.ndarray:
    if n_samples < 2:
        raise ValueError("sample requires at least two points.")
    start, end = _parameter_range(degree, knots)
    endpoint = closure == "open"
    return np.linspace(start, end, n_samples, endpoint=endpoint, dtype=float)


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

    @property
    def parameter_range(self) -> tuple[float, float]:
        return _parameter_range(self.degree, self.knots)

    def evaluate(self, parameter: float) -> np.ndarray:
        t = _normalize_parameter(
            parameter,
            degree=self.degree,
            knots=self.knots,
            closure=self.closure,
        )
        if self.closure == "open" and np.isclose(t, self.parameter_range[1]):
            return np.array(self.control_points[-1], dtype=float)
        return _evaluate_point(
            control_points=self.control_points,
            degree=self.degree,
            knots=self.knots,
            parameter=t,
        )

    def derivative(self, parameter: float) -> np.ndarray:
        t = _normalize_parameter(
            parameter,
            degree=self.degree,
            knots=self.knots,
            closure=self.closure,
        )
        return _evaluate_derivative(
            control_points=self.control_points,
            degree=self.degree,
            knots=self.knots,
            parameter=t,
        )

    def tangent(self, parameter: float) -> np.ndarray:
        derivative = self.derivative(parameter)
        norm = float(np.linalg.norm(derivative))
        if norm == 0.0:
            raise ValueError("tangent is undefined when derivative magnitude is zero.")
        return derivative / norm

    def sample(self, n_samples: int = 200) -> np.ndarray:
        params = _sample_parameters(
            degree=self.degree,
            knots=self.knots,
            closure=self.closure,
            n_samples=n_samples,
        )
        points = np.vstack([self.evaluate(float(t)) for t in params])
        if self.closure != "open":
            points = np.vstack([points, points[0]])
        return points


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

    @property
    def parameter_range(self) -> tuple[float, float]:
        return _parameter_range(self.degree, self.knots)

    def evaluate(self, parameter: float) -> np.ndarray:
        t = _normalize_parameter(
            parameter,
            degree=self.degree,
            knots=self.knots,
            closure=self.closure,
        )
        if self.closure == "open" and np.isclose(t, self.parameter_range[1]):
            return np.array(self.control_points[-1], dtype=float)
        return _evaluate_point(
            control_points=self.control_points,
            degree=self.degree,
            knots=self.knots,
            parameter=t,
        )

    def derivative(self, parameter: float) -> np.ndarray:
        t = _normalize_parameter(
            parameter,
            degree=self.degree,
            knots=self.knots,
            closure=self.closure,
        )
        return _evaluate_derivative(
            control_points=self.control_points,
            degree=self.degree,
            knots=self.knots,
            parameter=t,
        )

    def tangent(self, parameter: float) -> np.ndarray:
        derivative = self.derivative(parameter)
        norm = float(np.linalg.norm(derivative))
        if norm == 0.0:
            raise ValueError("tangent is undefined when derivative magnitude is zero.")
        return derivative / norm

    def sample(self, n_samples: int = 200) -> np.ndarray:
        params = _sample_parameters(
            degree=self.degree,
            knots=self.knots,
            closure=self.closure,
            n_samples=n_samples,
        )
        points = np.vstack([self.evaluate(float(t)) for t in params])
        if self.closure != "open":
            points = np.vstack([points, points[0]])
        return points


__all__ = ["BSpline2D", "BSpline3D", "ClosurePolicy"]
