from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

ParameterizationMethod = Literal["uniform", "chord_length", "centripetal"]


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


__all__ = ["ParameterizationMethod", "ParameterizationPolicyRecord"]
