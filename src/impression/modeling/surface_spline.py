from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .surface import _bspline_basis_functions, _find_bspline_span, _normalize_degree, _normalize_knot_vector, _validate_bspline_axis


@dataclass(frozen=True)
class SplineBasisEvaluationRecord:
    degree: int
    knots: tuple[float, ...]
    control_point_count: int
    parameter: float
    span: int
    basis: tuple[float, ...]

    @property
    def partition_error(self) -> float:
        return abs(float(sum(self.basis)) - 1.0)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "degree": self.degree,
            "knots": self.knots,
            "control_point_count": self.control_point_count,
            "parameter": self.parameter,
            "span": self.span,
            "basis": self.basis,
            "partition_error": self.partition_error,
        }


@dataclass(frozen=True)
class SplineKnotPolicyRecord:
    degree: int
    control_point_count: int
    knots: tuple[float, ...]
    parameter_range: tuple[float, float]
    clamped: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "degree": self.degree,
            "control_point_count": self.control_point_count,
            "knots": self.knots,
            "parameter_range": self.parameter_range,
            "clamped": self.clamped,
        }


@dataclass(frozen=True)
class LoftControlNetConstructionRecord:
    control_net: np.ndarray
    degree_u: int
    degree_v: int
    knots_u: tuple[float, ...]
    knots_v: tuple[float, ...]
    station_count: int
    profile_point_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_net", np.asarray(self.control_net, dtype=float))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "degree_u": self.degree_u,
            "degree_v": self.degree_v,
            "knots_u": self.knots_u,
            "knots_v": self.knots_v,
            "station_count": self.station_count,
            "profile_point_count": self.profile_point_count,
            "control_net_shape": tuple(int(value) for value in self.control_net.shape),
        }


def make_clamped_knot_vector(control_point_count: int, degree: int) -> tuple[float, ...]:
    """Build a deterministic open clamped knot vector on [0, 1]."""

    degree = _normalize_degree(degree, name="degree")
    count = int(control_point_count)
    if count <= degree:
        raise ValueError("control_point_count must be greater than degree.")
    interior_count = count - degree - 1
    interior = tuple(float(value) for value in np.linspace(0.0, 1.0, interior_count + 2)[1:-1])
    return (0.0,) * (degree + 1) + interior + (1.0,) * (degree + 1)


def validate_clamped_knot_vector(
    knots: Sequence[float],
    *,
    control_point_count: int,
    degree: int,
    name: str = "knots",
) -> SplineKnotPolicyRecord:
    """Validate spline knots and record the clamped policy expected by surface loft producers."""

    degree = _normalize_degree(degree, name="degree")
    normalized = _normalize_knot_vector(knots, name=name)
    parameter_range = _validate_bspline_axis(
        control_point_count=int(control_point_count),
        degree=degree,
        knots=normalized,
        name=name,
    )
    clamped = (
        all(value == normalized[0] for value in normalized[: degree + 1])
        and all(value == normalized[-1] for value in normalized[-degree - 1 :])
    )
    if not clamped:
        raise ValueError(f"{name} must be clamped at both ends.")
    return SplineKnotPolicyRecord(
        degree=degree,
        control_point_count=int(control_point_count),
        knots=normalized,
        parameter_range=parameter_range,
        clamped=True,
    )


def evaluate_bspline_basis(
    *,
    degree: int,
    knots: Sequence[float],
    control_point_count: int,
    parameter: float,
) -> SplineBasisEvaluationRecord:
    """Evaluate the non-zero B-spline basis functions at one parameter."""

    knot_record = validate_clamped_knot_vector(knots, control_point_count=control_point_count, degree=degree)
    start, end = knot_record.parameter_range
    u = float(parameter)
    if not np.isfinite(u):
        raise ValueError("parameter must be finite.")
    u = float(np.clip(u, start, end))
    span = _find_bspline_span(
        degree=knot_record.degree,
        knots=knot_record.knots,
        control_point_count=knot_record.control_point_count,
        parameter=u,
    )
    basis = tuple(float(value) for value in _bspline_basis_functions(span, u, knot_record.degree, knot_record.knots))
    return SplineBasisEvaluationRecord(
        degree=knot_record.degree,
        knots=knot_record.knots,
        control_point_count=knot_record.control_point_count,
        parameter=u,
        span=span,
        basis=basis,
    )


def build_loft_control_net(
    station_profiles: Sequence[Sequence[Sequence[float]] | np.ndarray],
    *,
    degree_u: int | None = None,
    degree_v: int | None = None,
) -> LoftControlNetConstructionRecord:
    """Construct a tensor-product control net from equal-size loft station profiles."""

    profiles = tuple(np.asarray(profile, dtype=float).reshape(-1, 3) for profile in station_profiles)
    if len(profiles) < 2:
        raise ValueError("station_profiles must contain at least two stations.")
    point_count = int(profiles[0].shape[0])
    if point_count < 2:
        raise ValueError("station profiles must contain at least two 3D points.")
    if any(profile.shape != (point_count, 3) for profile in profiles):
        raise ValueError("station profiles must have matching point counts.")
    if any(not np.all(np.isfinite(profile)) for profile in profiles):
        raise ValueError("station profiles must contain only finite values.")
    resolved_degree_u = min(3, len(profiles) - 1) if degree_u is None else _normalize_degree(degree_u, name="degree_u")
    resolved_degree_v = min(3, point_count - 1) if degree_v is None else _normalize_degree(degree_v, name="degree_v")
    control_net = np.stack(profiles, axis=0)
    knots_u = make_clamped_knot_vector(len(profiles), resolved_degree_u)
    knots_v = make_clamped_knot_vector(point_count, resolved_degree_v)
    return LoftControlNetConstructionRecord(
        control_net=control_net,
        degree_u=resolved_degree_u,
        degree_v=resolved_degree_v,
        knots_u=knots_u,
        knots_v=knots_v,
        station_count=len(profiles),
        profile_point_count=point_count,
    )


__all__ = [
    "LoftControlNetConstructionRecord",
    "SplineBasisEvaluationRecord",
    "SplineKnotPolicyRecord",
    "build_loft_control_net",
    "evaluate_bspline_basis",
    "make_clamped_knot_vector",
    "validate_clamped_knot_vector",
]
