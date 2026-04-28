from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

ParameterizationMethod = Literal["uniform", "chord_length", "centripetal"]
KnotCountStrategy = Literal["fixed"]
KnotPlacementMethod = Literal["uniform_internal", "average_parameter"]


def _require_ordered_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    if len(points) < 2:
        raise ValueError("parameter assignment requires at least two ordered samples.")
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError("sample points must be a sequence of 2D or 3D coordinates.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("sample points must be finite.")
    return arr


def _normalize_domain_start(value: float) -> float:
    start = float(value)
    if not np.isfinite(start):
        raise ValueError("domain_start must be finite.")
    return start


def _normalize_domain_end(value: float, *, start: float) -> float:
    end = float(value)
    if not np.isfinite(end):
        raise ValueError("domain_end must be finite.")
    if end <= start:
        raise ValueError("domain_end must be greater than domain_start.")
    return end


def _normalize_method(value: str) -> ParameterizationMethod:
    allowed: set[str] = {"uniform", "chord_length", "centripetal"}
    method = str(value)
    if method not in allowed:
        raise ValueError("method must be one of: uniform, chord_length, centripetal.")
    return method  # type: ignore[return-value]


def _normalize_knot_count_strategy(value: str) -> KnotCountStrategy:
    allowed: set[str] = {"fixed"}
    strategy = str(value)
    if strategy not in allowed:
        raise ValueError("strategy must be one of: fixed.")
    return strategy  # type: ignore[return-value]


def _normalize_positive_int(value: int, *, label: str) -> int:
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{label} must be >= 1.")
    return normalized


def _normalize_knot_placement_method(value: str) -> KnotPlacementMethod:
    allowed: set[str] = {"uniform_internal", "average_parameter"}
    method = str(value)
    if method not in allowed:
        raise ValueError("placement_method must be one of: uniform_internal, average_parameter.")
    return method  # type: ignore[return-value]


@dataclass(frozen=True)
class ParameterizationPolicyRecord:
    method: ParameterizationMethod = "chord_length"
    domain_start: float = 0.0
    domain_end: float = 1.0

    def __init__(
        self,
        method: ParameterizationMethod = "chord_length",
        domain_start: float = 0.0,
        domain_end: float = 1.0,
    ) -> None:
        normalized_method = _normalize_method(method)
        start = _normalize_domain_start(domain_start)
        end = _normalize_domain_end(domain_end, start=start)
        object.__setattr__(self, "method", normalized_method)
        object.__setattr__(self, "domain_start", start)
        object.__setattr__(self, "domain_end", end)

    def assign_parameters(self, samples: Sequence[Sequence[float]]) -> np.ndarray:
        points = _require_ordered_points(samples)
        base = self._base_parameters(points)
        return self.domain_start + (self.domain_end - self.domain_start) * base

    def _base_parameters(self, points: np.ndarray) -> np.ndarray:
        if self.method == "uniform":
            return np.linspace(0.0, 1.0, len(points), dtype=float)

        deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if self.method == "centripetal":
            deltas = np.sqrt(deltas)

        cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
        total = float(cumulative[-1])
        if total == 0.0:
            return np.linspace(0.0, 1.0, len(points), dtype=float)
        return cumulative / total


@dataclass(frozen=True)
class KnotCountPolicyRecord:
    strategy: KnotCountStrategy = "fixed"
    control_point_count: int = 4

    def __init__(
        self,
        strategy: KnotCountStrategy = "fixed",
        control_point_count: int = 4,
    ) -> None:
        normalized_strategy = _normalize_knot_count_strategy(strategy)
        normalized_count = _normalize_positive_int(
            control_point_count,
            label="control_point_count",
        )
        object.__setattr__(self, "strategy", normalized_strategy)
        object.__setattr__(self, "control_point_count", normalized_count)

    def resolve_control_point_count(self, *, sample_count: int, degree: int) -> int:
        sample_total = _normalize_positive_int(sample_count, label="sample_count")
        deg = _normalize_positive_int(degree, label="degree")
        if self.control_point_count <= deg:
            raise ValueError("control_point_count must be greater than degree.")
        if self.control_point_count > sample_total:
            raise ValueError("control_point_count must not exceed sample_count.")
        return self.control_point_count


@dataclass(frozen=True)
class KnotPlacementPolicyRecord:
    placement_method: KnotPlacementMethod = "uniform_internal"

    def __init__(self, placement_method: KnotPlacementMethod = "uniform_internal") -> None:
        object.__setattr__(
            self,
            "placement_method",
            _normalize_knot_placement_method(placement_method),
        )

    def build_knot_vector(
        self,
        parameters: Sequence[float],
        *,
        control_point_count: int,
        degree: int,
    ) -> tuple[float, ...]:
        params = np.asarray(parameters, dtype=float).reshape(-1)
        if params.size < 2:
            raise ValueError("parameters must contain at least two values.")
        if not np.all(np.isfinite(params)):
            raise ValueError("parameters must be finite.")
        if np.any(np.diff(params) < 0):
            raise ValueError("parameters must be nondecreasing.")

        ctrl_count = _normalize_positive_int(control_point_count, label="control_point_count")
        deg = _normalize_positive_int(degree, label="degree")
        if ctrl_count <= deg:
            raise ValueError("control_point_count must be greater than degree.")

        expected_knots = ctrl_count + deg + 1
        internal_count = expected_knots - 2 * (deg + 1)
        start = float(params[0])
        end = float(params[-1])
        if end <= start:
            raise ValueError("parameter domain must be increasing.")

        internal: np.ndarray
        if internal_count <= 0:
            internal = np.zeros((0,), dtype=float)
        elif self.placement_method == "uniform_internal":
            internal = np.linspace(start, end, internal_count + 2, dtype=float)[1:-1]
        else:
            internal = np.array(
                [
                    float(np.mean(params[j + 1 : j + deg + 1]))
                    for j in range(internal_count)
                ],
                dtype=float,
            )

        knots = np.concatenate(
            [
                np.full(deg + 1, start, dtype=float),
                internal,
                np.full(deg + 1, end, dtype=float),
            ]
        )
        return tuple(float(v) for v in knots)


@dataclass(frozen=True)
class FitConfigurationRecord:
    parameterization_policy: ParameterizationPolicyRecord
    knot_count_policy: KnotCountPolicyRecord
    knot_placement_policy: KnotPlacementPolicyRecord

    def __init__(
        self,
        *,
        parameterization_policy: ParameterizationPolicyRecord,
        knot_count_policy: KnotCountPolicyRecord,
        knot_placement_policy: KnotPlacementPolicyRecord,
    ) -> None:
        object.__setattr__(self, "parameterization_policy", parameterization_policy)
        object.__setattr__(self, "knot_count_policy", knot_count_policy)
        object.__setattr__(self, "knot_placement_policy", knot_placement_policy)

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "fit_configuration",
            self.parameterization_policy,
            self.knot_count_policy,
            self.knot_placement_policy,
        )


__all__ = [
    "FitConfigurationRecord",
    "KnotCountPolicyRecord",
    "KnotCountStrategy",
    "KnotPlacementMethod",
    "KnotPlacementPolicyRecord",
    "ParameterizationMethod",
    "ParameterizationPolicyRecord",
]
